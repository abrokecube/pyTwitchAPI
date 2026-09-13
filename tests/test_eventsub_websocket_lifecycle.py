import asyncio
import logging
import threading
from collections import deque

import pytest

from twitchAPI.eventsub.websocket import EventSubWebsocket, _validate_subscription_response
from twitchAPI.type import EventSubSubscriptionError


class _FakeTwitch:
    def has_required_auth(self, *_args, **_kwargs) -> bool:
        return True


def _new_eventsub(**attrs) -> EventSubWebsocket:
    client = object.__new__(EventSubWebsocket)
    client.logger = logging.getLogger('test.eventsub')
    client._twitch = _FakeTwitch()
    client._socket_thread = None
    client._socket_loop = None
    client._running = False
    client._startup_complete = False
    client._ready = False
    client._closing = False
    client._task_callback = lambda _task: None
    for key, value in attrs.items():
        setattr(client, key, value)
    return client


def _valid_subscription() -> dict:
    return {
        'id': 'sub-1',
        'status': 'enabled',
        'type': 'channel.chat.message',
        'version': '1',
        'cost': 0,
        'transport': {'method': 'websocket', 'session_id': 'session-1'},
    }


def _expect_invalid(item) -> None:
    with pytest.raises(EventSubSubscriptionError, match='invalid subscription response') as exc_info:
        _validate_subscription_response(
            status=202,
            payload={'data': [item]},
            requested_type='channel.chat.message',
            requested_version='1',
            session_id='session-1',
        )
    assert str(exc_info.value) == 'invalid subscription response'


@pytest.mark.parametrize(
    ('status', 'payload'),
    [
        (200, {'data': [_valid_subscription()]}),
        (202, {'data': []}),
        (202, {'data': [_valid_subscription(), _valid_subscription()]}),
        (202, {'data': [{**_valid_subscription(), 'status': 'pending'}]}),
        (202, {'data': [{**_valid_subscription(), 'type': 'stream.online'}]}),
        (202, {'data': [{**_valid_subscription(), 'version': '2'}]}),
        (202, {'data': [{**_valid_subscription(), 'transport': {'method': 'webhook'}}]}),
    ],
)
def test_subscription_success_shape_is_strict(status: int, payload: dict) -> None:
    with pytest.raises(EventSubSubscriptionError, match='invalid subscription response'):
        _validate_subscription_response(
            status=status,
            payload=payload,
            requested_type='channel.chat.message',
            requested_version='1',
            session_id='session-1',
        )


def test_subscription_success_shape_is_accepted() -> None:
    item = _valid_subscription()
    result = _validate_subscription_response(
        status=202,
        payload={'data': [item]},
        requested_type='channel.chat.message',
        requested_version='1',
        session_id='session-1',
    )
    assert result is item
    assert result['id'] == 'sub-1'


def test_subscription_rejects_mismatched_session_id() -> None:
    _expect_invalid({**_valid_subscription(), 'transport': {'method': 'websocket', 'session_id': 'session-2'}})


def test_subscription_rejects_missing_id() -> None:
    item = _valid_subscription()
    del item['id']
    _expect_invalid(item)


def test_subscription_rejects_empty_id() -> None:
    _expect_invalid({**_valid_subscription(), 'id': ''})


def test_subscription_rejects_non_integer_cost() -> None:
    _expect_invalid({**_valid_subscription(), 'cost': '0'})


def test_subscription_rejects_negative_cost() -> None:
    _expect_invalid({**_valid_subscription(), 'cost': -1})


def test_subscription_error_does_not_leak_sensitive_data() -> None:
    secret = 'super-secret-bearer-token'
    item = {**_valid_subscription(), 'status': 'pending', 'secret': secret}
    with pytest.raises(EventSubSubscriptionError) as exc_info:
        _validate_subscription_response(
            status=202,
            payload={'data': [item], 'token': secret},
            requested_type='channel.chat.message',
            requested_version='1',
            session_id='session-1',
        )
    message = str(exc_info.value)
    assert message == 'invalid subscription response'
    assert secret not in message
    assert 'session-1' not in message


def _notification(message_id, sub_id: str = 'sub-1') -> dict:
    return {
        'metadata': {'message_id': message_id, 'message_type': 'notification'},
        'payload': {'subscription': {'id': sub_id}, 'event': {}},
    }


async def _drain_tasks() -> None:
    await asyncio.sleep(0)
    await asyncio.sleep(0)


def test_duplicate_notification_id_delivers_once_and_history_is_bounded() -> None:
    delivered = []

    async def callback(event) -> None:
        delivered.append(event)

    async def scenario() -> None:
        client = _new_eventsub(
            _callbacks={'sub-1': {'id': 'sub-1', 'callback': callback, 'active': True, 'event': lambda **kw: kw}},
            _msg_id_history=deque(maxlen=2),
            _callback_loop=asyncio.get_running_loop(),
        )
        client._reset_timeout = lambda: None

        await client._handle_notification(_notification('m1'))
        await client._handle_notification(_notification('m1'))
        await _drain_tasks()
        assert len(delivered) == 1

        await client._handle_notification(_notification('m2'))
        await client._handle_notification(_notification('m3'))
        await _drain_tasks()
        assert len(delivered) == 3

        await client._handle_notification(_notification('m1'))
        await _drain_tasks()
        assert len(delivered) == 4

    asyncio.run(scenario())


def test_notification_without_message_id_is_always_delivered() -> None:
    delivered = []

    async def callback(event) -> None:
        delivered.append(event)

    async def scenario() -> None:
        client = _new_eventsub(
            _callbacks={'sub-1': {'id': 'sub-1', 'callback': callback, 'active': True, 'event': lambda **kw: kw}},
            _msg_id_history=deque(maxlen=1),
            _callback_loop=asyncio.get_running_loop(),
        )
        client._reset_timeout = lambda: None

        await client._handle_notification(_notification(None))
        await client._handle_notification(_notification(''))
        await _drain_tasks()
        assert len(delivered) == 2
        assert len(client._msg_id_history) == 0

    asyncio.run(scenario())


def test_eventsub_wait_closed_joins_publicly() -> None:
    client = object.__new__(EventSubWebsocket)
    thread = threading.Thread(target=lambda: None)
    thread.start()
    client._socket_thread = thread
    assert client.wait_closed(1.0) is True


def test_eventsub_wait_closed_before_start_returns_true() -> None:
    client = _new_eventsub()
    assert client.wait_closed() is True


def test_eventsub_wait_closed_from_socket_thread_raises() -> None:
    captured = {}

    def target() -> None:
        try:
            client.wait_closed(0.1)
        except BaseException as exc:  # noqa: BLE001 - inspect deliberate error
            captured['exc'] = exc

    client = _new_eventsub()
    thread = threading.Thread(target=target)
    client._socket_thread = thread
    thread.start()
    thread.join(1.0)
    assert isinstance(captured.get('exc'), RuntimeError)
    assert str(captured['exc']) == 'socket thread cannot wait for itself'


def test_eventsub_stop_is_idempotent() -> None:
    client = _new_eventsub()
    asyncio.run(client.stop())
    asyncio.run(client.stop())
    assert client.is_running is False
    assert client.is_ready is False


def test_eventsub_stop_without_socket_loop_reports_unsuccessful_termination() -> None:
    release = threading.Event()
    thread = threading.Thread(target=release.wait)
    thread.start()
    client = _new_eventsub(_running=True, _socket_thread=thread, _socket_loop=None)
    try:
        with pytest.raises(TimeoutError) as exc_info:
            asyncio.run(client.stop(timeout=0.05))
        assert str(exc_info.value) == 'Twitch socket thread did not stop'
    finally:
        release.set()
        thread.join(1.0)
    assert client.is_running is False
    assert client.is_ready is False


def test_eventsub_partial_start_cleanup() -> None:
    client = _new_eventsub()
    client._run_socket = lambda: None  # socket thread dies before startup completes
    with pytest.raises(RuntimeError):
        client.start()
    assert client.is_running is False
    assert client.is_ready is False
    assert client.wait_closed(0.1) is True


def test_eventsub_stop_clears_socket_loop_on_success() -> None:
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever)
    loop_thread.start()
    release = threading.Event()
    helper_started = threading.Event()

    def socket_helper() -> None:
        helper_started.set()
        release.wait(2.0)

    helper = threading.Thread(target=socket_helper)
    helper.start()
    assert helper_started.wait(1.0)

    async def short_stop() -> None:
        release.set()

    client = _new_eventsub(_running=True, _socket_loop=loop, _socket_thread=helper)
    client._stop = short_stop
    try:
        asyncio.run(client.stop(timeout=1.0))
        assert client._socket_loop is None
        assert client._socket_thread is None
        assert client.is_running is False
    finally:
        release.set()
        helper.join(1.0)
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(1.0)
        loop.close()
    assert helper.is_alive() is False
    assert loop_thread.is_alive() is False


def test_eventsub_stop_from_socket_thread_raises() -> None:
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever)
    loop_thread.start()
    client = _new_eventsub(_running=True, _socket_loop=loop, _socket_thread=loop_thread)
    try:
        future = asyncio.run_coroutine_threadsafe(client.stop(timeout=0.05), loop)
        with pytest.raises(RuntimeError) as exc_info:
            future.result(2.0)
        assert str(exc_info.value) == 'socket thread cannot stop itself'
    finally:
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(1.0)
        loop.close()
    assert loop_thread.is_alive() is False
