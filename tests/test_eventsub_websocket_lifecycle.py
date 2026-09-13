import asyncio
import datetime
import json
import logging
import threading
import time
from collections import deque
from types import SimpleNamespace

import aiohttp
import pytest

from twitchAPI.eventsub import websocket as eventsub_websocket
from twitchAPI.eventsub.websocket import EventSubWebsocket, Session, _validate_subscription_response
from twitchAPI.type import ConnectionState, EventSubSubscriptionError


class _FakeTwitch:
    session_timeout = 30
    base_url = 'https://api.twitch.tv/helix/'

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
    client._connection = None
    client._session = None
    client._callback_loop = None
    client._configured_callback_loop = None
    client._state_lock = threading.RLock()
    client.connection_url = 'wss://example.invalid/ws'
    client._is_reconnecting = False
    client._reconnect_timeout = None
    client._task_callback = lambda _task: None
    if 'state_change_handler' in attrs:
        client._state_change_handler = attrs.pop('state_change_handler')
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
        (202, {}),
        (202, {'data': 'not-a-list'}),
        (202, {'data': ['not-a-dict']}),
        (202, {'data': [None]}),
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


class _FakeHttpResponse:
    def __init__(self, status: int, payload: dict) -> None:
        self.status = status
        self._payload = payload

    async def json(self) -> dict:
        return self._payload


class _FakeClientSession:
    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self) -> '_FakeClientSession':
        return self

    async def __aexit__(self, *_exc_info) -> bool:
        return False


class _SomeEvent:
    pass


def test_malformed_subscribe_response_registers_no_callback(monkeypatch) -> None:
    async def callback(event) -> None:
        pass

    async def fake_post_request(session, url, data=None):
        malformed = {**_valid_subscription(), 'status': 'pending'}
        return _FakeHttpResponse(202, {'data': [malformed]})

    monkeypatch.setattr(eventsub_websocket, 'ClientSession', _FakeClientSession)

    async def scenario() -> None:
        client = _new_eventsub(
            _callbacks={},
            _active_subscriptions={},
            active_session=SimpleNamespace(id='session-1'),
            subscription_url='https://example.invalid/',
        )
        client._api_post_request = fake_post_request

        with pytest.raises(EventSubSubscriptionError, match='invalid subscription response'):
            await client._subscribe(
                'channel.chat.message',
                '1',
                {'broadcaster_user_id': '1', 'user_id': '2'},
                callback,
                _SomeEvent,
            )
        assert client._callbacks == {}
        assert client._active_subscriptions == {}

    asyncio.run(scenario())


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


# --- connection state projection ---

class _FakeConnection:
    def __init__(self) -> None:
        self.closed = False
        self._queue: asyncio.Queue = asyncio.Queue()

    async def close(self) -> None:
        if not self.closed:
            self.closed = True
            # emulate aiohttp waking a pending receive with a CLOSING message
            self._queue.put_nowait(_FakeWSMessage(aiohttp.WSMsgType.CLOSING, ''))

    async def receive(self, timeout=None):
        return await self._queue.get()

    def push(self, message) -> None:
        self._queue.put_nowait(message)

    def exception(self):
        return None


class _FakeWebsocketSession:
    async def ws_connect(self, _url: str) -> _FakeConnection:
        return _FakeConnection()


class _HandoverSession:
    def __init__(self, connection: _FakeConnection) -> None:
        self._connection = connection

    async def ws_connect(self, _url: str) -> _FakeConnection:
        return self._connection


class _FakeWSMessage:
    def __init__(self, msg_type, data: str) -> None:
        self.type = msg_type
        self.data = data

    def json(self) -> dict:
        return json.loads(self.data)


def _welcome(session_id: str, keepalive_timeout_seconds: int = 30) -> dict:
    return {
        'metadata': {'message_type': 'session_welcome'},
        'payload': {
            'session': {
                'id': session_id,
                'keepalive_timeout_seconds': keepalive_timeout_seconds,
                'status': 'connected',
                'reconnect_url': 'wss://example.invalid/reconnect',
            }
        },
    }


def test_eventsub_connection_state_cycle_is_ordered() -> None:
    states = []
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()
    release = threading.Event()
    helper_started = threading.Event()

    def fake_run_socket() -> None:
        client._socket_loop = loop
        client._callback_loop = loop
        client._startup_complete = True
        helper_started.set()
        release.wait(2.0)

    async def short_stop() -> None:
        release.set()

    client = _new_eventsub(
        state_change_handler=states.append,
        _session=_FakeWebsocketSession(),
    )
    client._run_socket = fake_run_socket
    client._stop = short_stop
    try:
        client.start()
        assert helper_started.wait(1.0)

        async def first_welcome() -> None:
            await client._handle_welcome(_welcome('sess-1'))

        asyncio.run(first_welcome())

        async def reconnect() -> None:
            await client._connect(is_startup=False)

        asyncio.run(reconnect())

        async def second_welcome() -> None:
            client._is_reconnecting = True
            await client._handle_welcome(_welcome('sess-2'))

        asyncio.run(second_welcome())

        asyncio.run(client.stop(timeout=1.0))
    finally:
        release.set()
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(1.0)
        loop.close()

    assert loop_thread.is_alive() is False
    assert states == [
        ConnectionState.STARTING,
        ConnectionState.READY,
        ConnectionState.RECONNECTING,
        ConnectionState.READY,
        ConnectionState.STOPPING,
        ConnectionState.STOPPED,
    ]


def test_eventsub_partial_start_reports_failed() -> None:
    states = []
    client = _new_eventsub(state_change_handler=states.append)
    client._run_socket = lambda: None
    with pytest.raises(RuntimeError):
        client.start()
    assert states == [ConnectionState.STARTING, ConnectionState.FAILED]


def test_eventsub_stop_reports_stopping_and_stopped() -> None:
    states = []
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
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

    client = _new_eventsub(
        state_change_handler=states.append,
        _running=True,
        _socket_loop=loop,
        _socket_thread=helper,
    )
    client._stop = short_stop
    try:
        asyncio.run(client.stop(timeout=1.0))
    finally:
        release.set()
        helper.join(1.0)
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(1.0)
        loop.close()

    assert states == [ConnectionState.STOPPING, ConnectionState.STOPPED]


def test_eventsub_state_payloads_are_enum_only() -> None:
    captured = []
    client = _new_eventsub(state_change_handler=captured.append)
    client._set_connection_state(ConnectionState.STARTING)
    client._set_connection_state(ConnectionState.READY)
    assert captured == [ConnectionState.STARTING, ConnectionState.READY]
    assert all(isinstance(state, ConnectionState) for state in captured)
    text = repr(captured)
    for secret in ('wss://', 'sess-1', 'Bearer', 'Authorization', 'token'):
        assert secret not in text


def test_eventsub_state_handler_exception_is_swallowed() -> None:
    def handler(_state) -> None:
        raise RuntimeError('handler boom token=super-secret')

    client = _new_eventsub(state_change_handler=handler)
    client._set_connection_state(ConnectionState.STARTING)
    client._set_connection_state(ConnectionState.READY)
    assert client.connection_state is ConnectionState.READY


def _reconnect_request() -> dict:
    return {
        'metadata': {'message_type': 'session_reconnect'},
        'payload': {
            'session': {
                'id': 'sess-old',
                'keepalive_timeout_seconds': 30,
                'status': 'reconnect',
                'reconnect_url': 'wss://example.invalid/reconnect',
            }
        },
    }


def test_eventsub_state_updates_are_atomic_and_deadlock_free() -> None:
    entered = threading.Event()
    release = threading.Event()
    calls = []
    calls_lock = threading.Lock()

    def handler(state) -> None:
        with calls_lock:
            calls.append(state)
        if state is ConnectionState.READY:
            entered.set()
            release.wait(2.0)

    client = _new_eventsub(state_change_handler=handler)
    first = threading.Thread(target=lambda: client._set_connection_state(ConnectionState.READY))
    second = threading.Thread(target=lambda: client._set_connection_state(ConnectionState.FAILED))
    try:
        first.start()
        assert entered.wait(1.0) is True
        assert client.connection_state is ConnectionState.READY
        # The handler runs outside the state lock, so a concurrent transition can update the
        # stored state and be notified without waiting for the blocked READY handler.
        second.start()
        second.join(1.0)
        assert second.is_alive() is False
        assert client.connection_state is ConnectionState.FAILED
        with calls_lock:
            assert calls == [ConnectionState.READY, ConnectionState.FAILED]
        # identical consecutive states must not re-notify
        client._set_connection_state(ConnectionState.FAILED)
        with calls_lock:
            assert calls.count(ConnectionState.FAILED) == 1
    finally:
        release.set()
        first.join(1.0)
        second.join(1.0)
    assert client.connection_state is ConnectionState.FAILED


def test_eventsub_state_handler_calling_stop_does_not_deadlock() -> None:
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()
    socket_release = threading.Event()
    reentered = threading.Event()
    calls = []
    errors = []

    async def fake_stop() -> None:
        socket_release.set()

    def socket_body() -> None:
        socket_release.wait(2.0)
        # emulate _run_socket's finally surfacing the terminal state
        client._set_connection_state(ConnectionState.STOPPED)

    socket_thread = threading.Thread(target=socket_body, daemon=True)

    def handler(state) -> None:
        calls.append(state)
        if state is ConnectionState.READY and not reentered.is_set():
            reentered.set()
            try:
                asyncio.run(client.stop(timeout=1.0))
            except BaseException as exc:  # noqa: BLE001 - recorded for the regression assertion
                errors.append(exc)

    client = _new_eventsub(
        state_change_handler=handler,
        _running=True,
        _socket_loop=loop,
        _socket_thread=socket_thread,
    )
    client._stop = fake_stop
    socket_thread.start()
    try:
        client._set_connection_state(ConnectionState.READY)
        socket_thread.join(2.0)
    finally:
        socket_release.set()
        socket_thread.join(1.0)
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(1.0)
        loop.close()
    assert errors == []
    assert calls == [ConnectionState.READY, ConnectionState.STOPPING, ConnectionState.STOPPED]


def _resubscription() -> dict:
    return {
        'sub_type': 'channel.chat.message',
        'sub_version': '1',
        'condition': {'broadcaster_user_id': '1', 'user_id': '2'},
        'callback': lambda _event: None,
        'event': lambda **kw: kw,
    }


def test_resubscribe_propagates_cancellation() -> None:
    attempted = []

    async def cancelled_subscribe(sub_type, sub_version, condition, callback, event, is_batching_enabled=None):
        attempted.append(sub_type)
        raise asyncio.CancelledError()

    async def scenario() -> None:
        client = _new_eventsub(_active_subscriptions={'sub-1': _resubscription()})
        client._subscribe = cancelled_subscribe
        with pytest.raises(asyncio.CancelledError):
            await client._resubscribe()

    asyncio.run(scenario())
    assert attempted == ['channel.chat.message']


def test_resubscribe_restores_subscriptions_on_real_error() -> None:
    async def failing_subscribe(sub_type, sub_version, condition, callback, event, is_batching_enabled=None):
        raise RuntimeError('boom')

    async def scenario() -> None:
        client = _new_eventsub(_active_subscriptions={'sub-1': _resubscription()})
        client._subscribe = failing_subscribe
        await client._resubscribe()
        assert list(client._active_subscriptions) == ['sub-1']

    asyncio.run(scenario())


async def _wait_for_state(client, state: ConnectionState, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while client.connection_state is not state:
        if time.monotonic() > deadline:
            raise AssertionError(f'timed out waiting for {state}, currently {client.connection_state}')
        await asyncio.sleep(0.005)


async def _wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError('timed out waiting for condition')
        await asyncio.sleep(0.005)


def test_eventsub_reconnect_handover_emits_ready_and_dedupes_across_swap() -> None:
    delivered = []
    states = []
    resubscribed = []

    async def callback(event) -> None:
        delivered.append(event)

    async def scenario() -> None:
        old_connection = _FakeConnection()
        new_connection = _FakeConnection()
        subscription = {
            'sub_type': 'channel.chat.message',
            'sub_version': '1',
            'condition': {'broadcaster_user_id': '1', 'user_id': '2'},
            'callback': callback,
            'event': lambda **kw: kw,
        }
        client = _new_eventsub(
            state_change_handler=states.append,
            _callbacks={'sub-1': {'id': 'sub-1', 'callback': callback, 'active': True, 'event': lambda **kw: kw}},
            _msg_id_history=deque(maxlen=5),
            _callback_loop=asyncio.get_running_loop(),
            _connection=old_connection,
            active_session=Session(
                id='sess-old',
                keepalive_timeout_seconds=30,
                status='connected',
                reconnect_url=None,
            ),
            _session=_HandoverSession(new_connection),
            _active_subscriptions={'sub-1': subscription},
            _is_reconnecting=False,
            _reset_timeout=lambda: None,
            _running=True,
        )

        async def fake_subscribe(sub_type, sub_version, condition, callback, event, is_batching_enabled=None):
            resubscribed.append((sub_type, client.active_session.id))
            client._active_subscriptions['sub-1'] = {
                'sub_type': sub_type,
                'sub_version': sub_version,
                'condition': condition,
                'callback': callback,
                'event': event,
            }
            return 'sub-1'

        client._subscribe = fake_subscribe

        client._set_connection_state(ConnectionState.STARTING)
        receive_task = asyncio.ensure_future(client._task_receive())
        try:
            old_connection.push(_FakeWSMessage(aiohttp.WSMsgType.TEXT, json.dumps(_welcome('sess-old'))))
            await _wait_for_state(client, ConnectionState.READY)

            old_connection.push(
                _FakeWSMessage(aiohttp.WSMsgType.TEXT, json.dumps(_notification('m-handover')))
            )
            await _wait_until(lambda: len(delivered) == 1)

            old_connection.push(
                _FakeWSMessage(aiohttp.WSMsgType.TEXT, json.dumps(_reconnect_request()))
            )
            await _wait_for_state(client, ConnectionState.RECONNECTING)

            # the replacement welcome is delayed: the old socket must stay in place and consumable
            assert client._connection is old_connection
            assert client._reconnect is None

            new_connection.push(_FakeWSMessage(aiohttp.WSMsgType.TEXT, json.dumps(_welcome('sess-new'))))
            await _wait_for_state(client, ConnectionState.READY)

            assert client.active_session.id == 'sess-new'
            assert client._connection is new_connection
            assert resubscribed == [('channel.chat.message', 'sess-new')]

            # the replacement redelivers the same message id -> exactly one callback total
            new_connection.push(
                _FakeWSMessage(aiohttp.WSMsgType.TEXT, json.dumps(_notification('m-handover')))
            )
            await _drain_tasks()
            await asyncio.sleep(0.01)
            assert len(delivered) == 1
        finally:
            receive_task.cancel()
            await asyncio.gather(receive_task, return_exceptions=True)

    asyncio.run(scenario())
    assert states == [
        ConnectionState.STARTING,
        ConnectionState.READY,
        ConnectionState.RECONNECTING,
        ConnectionState.READY,
    ]


# --- is_ready projection ---

def test_eventsub_is_ready_starts_false() -> None:
    client = _new_eventsub()
    assert client.is_ready is False
    assert client.connection_state is ConnectionState.STOPPED


def test_eventsub_first_welcome_sets_is_ready() -> None:
    client = _new_eventsub()
    asyncio.run(client._handle_welcome(_welcome('sess-1')))
    assert client.connection_state is ConnectionState.READY
    assert client.is_ready is True


def test_eventsub_connect_reconnect_entry_clears_is_ready() -> None:
    client = _new_eventsub(_session=_FakeWebsocketSession())
    client._set_connection_state(ConnectionState.READY)
    assert client.is_ready is True
    asyncio.run(client._connect(is_startup=False))
    assert client.connection_state is ConnectionState.RECONNECTING
    assert client.is_ready is False


def test_eventsub_handle_reconnect_clears_then_handover_sets_is_ready() -> None:
    async def scenario() -> None:
        old_connection = _FakeConnection()
        new_connection = _FakeConnection()
        client = _new_eventsub(
            _connection=old_connection,
            active_session=Session(
                id='sess-old',
                keepalive_timeout_seconds=30,
                status='connected',
                reconnect_url=None,
            ),
            _session=_HandoverSession(new_connection),
            _callback_loop=asyncio.get_running_loop(),
            _reset_timeout=lambda: None,
            _active_subscriptions={},
            _callbacks={},
            _running=True,
        )
        client._set_connection_state(ConnectionState.READY)
        assert client.is_ready is True
        receive_task = asyncio.ensure_future(client._task_receive())
        try:
            old_connection.push(_FakeWSMessage(aiohttp.WSMsgType.TEXT, json.dumps(_reconnect_request())))
            await _wait_for_state(client, ConnectionState.RECONNECTING)
            assert client.is_ready is False

            new_connection.push(_FakeWSMessage(aiohttp.WSMsgType.TEXT, json.dumps(_welcome('sess-new'))))
            await _wait_for_state(client, ConnectionState.READY)
            assert client.is_ready is True
        finally:
            receive_task.cancel()
            await asyncio.gather(receive_task, return_exceptions=True)

    asyncio.run(scenario())


def test_eventsub_stop_and_startup_failure_clear_is_ready() -> None:
    stopped = _new_eventsub(_running=True, _socket_thread=None, _socket_loop=None)
    stopped._set_connection_state(ConnectionState.READY)
    assert stopped.is_ready is True
    asyncio.run(stopped.stop(timeout=1.0))
    assert stopped.is_ready is False

    failed = _new_eventsub()
    failed._set_connection_state(ConnectionState.READY)
    assert failed.is_ready is True
    failed._run_socket = lambda: None
    with pytest.raises(RuntimeError):
        failed.start()
    assert failed.is_ready is False


async def _noop(*_args, **_kwargs) -> None:
    return None


async def _drain_cancelled(tasks) -> None:
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


_ALL_STATES = (
    ConnectionState.STARTING,
    ConnectionState.READY,
    ConnectionState.RECONNECTING,
    ConnectionState.FAILED,
    ConnectionState.STOPPING,
    ConnectionState.STOPPED,
)


def test_eventsub_is_ready_is_derived_from_connection_state() -> None:
    client = _new_eventsub()
    for state in _ALL_STATES:
        client._set_connection_state(state)
        assert client.connection_state is state
        assert client.is_ready is (state is ConnectionState.READY)


def test_eventsub_stopping_handler_observes_not_ready() -> None:
    observed = []

    def handler(state) -> None:
        observed.append((state, client.is_ready))

    client = _new_eventsub(state_change_handler=handler, _running=True)
    client._set_connection_state(ConnectionState.READY)
    assert client.is_ready is True
    asyncio.run(client.stop(timeout=0.1))
    assert (ConnectionState.STOPPING, False) in observed
    assert (ConnectionState.STOPPED, False) in observed


def test_eventsub_socket_exit_without_closing_reports_failed_and_not_ready() -> None:
    states = []
    client = _new_eventsub(state_change_handler=states.append, _closing=False, _ready=True)
    client._connect = _noop
    client._task_receive = _noop
    client._task_reconnect_handler = _noop
    client._keep_loop_alive = _noop
    client._run_socket()
    socket_loop = client._socket_loop
    try:
        assert states == [ConnectionState.FAILED]
        assert client.connection_state is ConnectionState.FAILED
        assert client.is_ready is False
    finally:
        socket_loop.run_until_complete(_drain_cancelled(client._tasks))
        socket_loop.close()
        asyncio.set_event_loop(None)


# --- keepalive-expiry reconnect handling (real _task_reconnect_handler) ---

class _RecordingWebsocketSession:
    def __init__(self) -> None:
        self.calls: list = []
        self.connection = _FakeConnection()

    async def ws_connect(self, url: str) -> _FakeConnection:
        self.calls.append(url)
        return self.connection


def _expired_deadline() -> datetime.datetime:
    return datetime.datetime.now() - datetime.timedelta(seconds=1)


def _future_deadline() -> datetime.datetime:
    return datetime.datetime.now() + datetime.timedelta(seconds=30)


def _start_reconnect_watch(session, **attrs):
    """Build a client whose (real) ``_connect`` is wrapped to record ``is_startup``."""
    callbacks = attrs.pop('_callbacks', {})
    active = attrs.pop('_active_subscriptions', {})
    client = _new_eventsub(
        _session=session,
        _callbacks=callbacks,
        _active_subscriptions=active,
        _callback_loop=asyncio.get_running_loop(),
        _reset_timeout=lambda: None,
        **attrs,
    )
    connect_calls: list = []
    real_connect = client._connect

    async def recording_connect(is_startup: bool = False):
        connect_calls.append(is_startup)
        await real_connect(is_startup=is_startup)

    client._connect = recording_connect
    return client, connect_calls


def test_keepalive_future_deadline_does_not_reconnect_early() -> None:
    async def scenario() -> None:
        session = _RecordingWebsocketSession()
        client, connect_calls = _start_reconnect_watch(session)
        client._reconnect_timeout = _future_deadline()
        task = asyncio.ensure_future(client._task_reconnect_handler())
        try:
            await asyncio.sleep(0.15)
            assert connect_calls == []
            assert client.connection_state is not ConnectionState.RECONNECTING
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_keepalive_expired_deadline_reconnects_with_is_startup_false() -> None:
    async def scenario() -> None:
        session = _RecordingWebsocketSession()
        client, connect_calls = _start_reconnect_watch(session)
        client._set_connection_state(ConnectionState.READY)
        client._reconnect_timeout = _expired_deadline()
        task = asyncio.ensure_future(client._task_reconnect_handler())
        try:
            await _wait_until(lambda: len(connect_calls) == 1)
            assert connect_calls == [False]
            assert session.calls == ['wss://example.invalid/ws']
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_keepalive_expired_deadline_exposes_reconnecting_and_not_ready() -> None:
    async def scenario() -> None:
        session = _RecordingWebsocketSession()
        client, connect_calls = _start_reconnect_watch(session)
        client._set_connection_state(ConnectionState.READY)
        assert client.is_ready is True
        client._reconnect_timeout = _expired_deadline()
        task = asyncio.ensure_future(client._task_reconnect_handler())
        try:
            await _wait_for_state(client, ConnectionState.RECONNECTING)
            assert connect_calls == [False]
            assert client.is_ready is False
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_keepalive_reconnect_clears_deadline_and_does_not_storm() -> None:
    async def scenario() -> None:
        session = _RecordingWebsocketSession()
        client, connect_calls = _start_reconnect_watch(session)
        client._reconnect_timeout = _expired_deadline()
        task = asyncio.ensure_future(client._task_reconnect_handler())
        try:
            await _wait_until(lambda: len(connect_calls) == 1)
            assert client._reconnect_timeout is None
            await asyncio.sleep(0.3)
            assert connect_calls == [False]
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_keepalive_reconnect_handler_exits_cleanly_on_cancel() -> None:
    async def scenario() -> None:
        session = _RecordingWebsocketSession()
        client, _ = _start_reconnect_watch(session)
        task = asyncio.ensure_future(client._task_reconnect_handler())
        await asyncio.sleep(0.15)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert task.done() is True
        assert task.cancelled() is False
        assert task.exception() is None

    asyncio.run(scenario())


def test_keepalive_reconnect_handler_exits_cleanly_when_stopping() -> None:
    async def scenario() -> None:
        session = _RecordingWebsocketSession()
        client, _ = _start_reconnect_watch(session)
        task = asyncio.ensure_future(client._task_reconnect_handler())
        await asyncio.sleep(0.15)
        client._closing = True
        await asyncio.wait_for(task, timeout=1.0)
        assert task.done() is True
        assert task.cancelled() is False

    asyncio.run(scenario())


def test_keepalive_reconnect_recovers_subscriptions_and_returns_ready() -> None:
    async def scenario() -> None:
        session = _RecordingWebsocketSession()
        subscription = {
            'sub_type': 'channel.chat.message',
            'sub_version': '1',
            'condition': {'broadcaster_user_id': '1', 'user_id': '2'},
            'callback': lambda _event: None,
            'event': lambda **kw: kw,
        }
        resubscribed: list = []
        client, connect_calls = _start_reconnect_watch(
            session,
            _active_subscriptions={'sub-1': subscription},
            _callbacks={'sub-1': {'id': 'sub-1', 'callback': lambda _e: None, 'active': True, 'event': lambda **kw: kw}},
        )
        client._set_connection_state(ConnectionState.READY)
        client._reconnect_timeout = _expired_deadline()

        async def fake_subscribe(sub_type, sub_version, condition, callback, event, is_batching_enabled=None):
            resubscribed.append(sub_type)
            client._active_subscriptions['sub-1'] = {
                'sub_type': sub_type,
                'sub_version': sub_version,
                'condition': condition,
                'callback': callback,
                'event': event,
            }
            return 'sub-1'

        client._subscribe = fake_subscribe
        task = asyncio.ensure_future(client._task_reconnect_handler())
        try:
            await _wait_for_state(client, ConnectionState.RECONNECTING)
            assert connect_calls == [False]
            assert client.is_ready is False

            await client._handle_welcome(_welcome('sess-new'))
            assert client.connection_state is ConnectionState.READY
            assert client.is_ready is True
            assert client._is_reconnecting is False
            assert resubscribed == ['channel.chat.message']
            assert list(client._active_subscriptions) == ['sub-1']
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_handle_keepalive_resets_reconnect_deadline_to_future() -> None:
    async def scenario() -> None:
        client = _new_eventsub(
            active_session=Session(
                id='sess-1',
                keepalive_timeout_seconds=30,
                status='connected',
                reconnect_url=None,
            ),
        )
        client._reconnect_timeout = _expired_deadline()
        await client._handle_keepalive({})
        assert client._reconnect_timeout is not None
        assert client._reconnect_timeout > datetime.datetime.now()

    asyncio.run(scenario())
