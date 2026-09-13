import asyncio
import logging
import threading

import pytest

from twitchAPI.chat import Chat
from twitchAPI.type import ConnectionState


class _FakeTwitch:
    def has_required_auth(self, *_args, **_kwargs) -> bool:
        return True


def _new_chat(**attrs) -> Chat:
    chat = object.__new__(Chat)
    chat.logger = logging.getLogger('test.chat')
    chat.twitch = _FakeTwitch()
    chat.username = 'testbot'
    chat.no_shared_chat_messages = True
    chat._Chat__socket_thread = None
    chat._Chat__socket_loop = None
    chat._Chat__running = False
    chat._Chat__startup_complete = False
    chat._ready = False
    chat._closing = False
    chat._join_target = []
    chat._event_handler = {}
    chat._callback_loop = None
    chat._configured_callback_loop = None
    chat._state_lock = threading.RLock()
    if 'state_change_handler' in attrs:
        chat._state_change_handler = attrs.pop('state_change_handler')
    for key, value in attrs.items():
        setattr(chat, key, value)
    return chat


def test_chat_wait_closed_times_out_without_private_access() -> None:
    release = threading.Event()
    thread = threading.Thread(target=release.wait)
    thread.start()
    chat = object.__new__(Chat)
    chat._Chat__socket_thread = thread
    try:
        assert chat.wait_closed(0.01) is False
    finally:
        release.set()
        thread.join(1.0)


def test_chat_wait_closed_before_start_returns_true() -> None:
    chat = _new_chat()
    assert chat.wait_closed() is True


def test_chat_wait_closed_from_socket_thread_raises() -> None:
    captured = {}

    def target() -> None:
        try:
            chat.wait_closed(0.1)
        except BaseException as exc:  # noqa: BLE001 - inspect deliberate error
            captured['exc'] = exc

    chat = _new_chat()
    thread = threading.Thread(target=target)
    chat._Chat__socket_thread = thread
    thread.start()
    thread.join(1.0)
    assert isinstance(captured.get('exc'), RuntimeError)
    assert str(captured['exc']) == 'socket thread cannot wait for itself'


def test_chat_stop_is_idempotent() -> None:
    chat = _new_chat()
    chat.stop()
    chat.stop()
    assert chat.is_running is False
    assert chat.is_ready is False


def test_chat_stop_before_start_is_a_noop() -> None:
    chat = _new_chat()
    chat.stop()
    assert chat.wait_closed(0.1) is True


def test_chat_partial_start_cleanup() -> None:
    chat = _new_chat()
    chat._Chat__run_socket = lambda: None  # socket thread dies before startup completes
    with pytest.raises(RuntimeError):
        chat.start()
    assert chat.is_running is False
    assert chat.is_ready is False
    assert chat.wait_closed(0.1) is True


def test_chat_stop_bounds_socket_stop_future() -> None:
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever)
    loop_thread.start()
    try:
        async def never_ending_stop() -> None:
            await asyncio.Event().wait()

        chat = _new_chat(_Chat__running=True, _Chat__socket_loop=loop, _Chat__socket_thread=None)
        chat._stop = never_ending_stop
        with pytest.raises(TimeoutError) as exc_info:
            chat.stop(timeout=0.01)
        assert type(exc_info.value) is TimeoutError
        assert str(exc_info.value) == 'Twitch socket thread did not stop'
        assert chat.is_running is False
        assert chat.is_ready is False
    finally:
        async def cancel_pending() -> None:
            current = asyncio.current_task()
            pending = [task for task in asyncio.all_tasks() if task is not current]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        shutdown = asyncio.run_coroutine_threadsafe(cancel_pending(), loop)
        try:
            shutdown.result(1.0)
        finally:
            loop.call_soon_threadsafe(loop.stop)
            loop_thread.join(1.0)
            loop.close()
    assert loop_thread.is_alive() is False


def test_chat_stop_clears_socket_loop_on_success() -> None:
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

    chat = _new_chat(_Chat__running=True, _Chat__socket_loop=loop, _Chat__socket_thread=helper)
    chat._stop = short_stop
    try:
        chat.stop(timeout=1.0)
        assert chat._Chat__socket_loop is None
        assert chat._Chat__socket_thread is None
        assert chat.is_running is False
    finally:
        release.set()
        helper.join(1.0)
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(1.0)
        loop.close()
    assert helper.is_alive() is False
    assert loop_thread.is_alive() is False


def test_chat_stop_from_socket_thread_raises() -> None:
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever)
    loop_thread.start()
    captured = {}
    done = threading.Event()

    def invoke_stop() -> None:
        try:
            chat.stop(timeout=0.05)
        except BaseException as exc:  # noqa: BLE001 - inspect deliberate error
            captured['exc'] = exc
        finally:
            done.set()

    chat = _new_chat(_Chat__running=True, _Chat__socket_loop=loop, _Chat__socket_thread=loop_thread)
    try:
        loop.call_soon_threadsafe(invoke_stop)
        assert done.wait(2.0)
        assert isinstance(captured.get('exc'), RuntimeError)
        assert str(captured['exc']) == 'socket thread cannot stop itself'
    finally:
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(1.0)
        loop.close()
    assert loop_thread.is_alive() is False


# --- connection state projection ---

def test_chat_connection_state_start_ready_stop() -> None:
    states = []
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()
    release = threading.Event()
    helper_started = threading.Event()

    def fake_run_socket() -> None:
        chat._Chat__socket_loop = loop
        chat._callback_loop = loop
        chat._Chat__startup_complete = True
        helper_started.set()
        release.wait(2.0)

    async def short_stop() -> None:
        release.set()

    chat = _new_chat(state_change_handler=states.append)
    chat._Chat__run_socket = fake_run_socket
    chat._stop = short_stop
    try:
        chat.start()
        assert helper_started.wait(1.0)

        async def ready() -> None:
            await chat._handle_ready({'parameters': None, 'tags': {}})

        asyncio.run(ready())
        chat.stop(timeout=1.0)
    finally:
        release.set()
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(1.0)
        loop.close()

    assert loop_thread.is_alive() is False
    assert states == [
        ConnectionState.STARTING,
        ConnectionState.READY,
        ConnectionState.STOPPING,
        ConnectionState.STOPPED,
    ]


def test_chat_partial_start_reports_failed() -> None:
    states = []
    chat = _new_chat(state_change_handler=states.append)
    chat._Chat__run_socket = lambda: None  # socket thread dies before startup completes
    with pytest.raises(RuntimeError):
        chat.start()
    assert states == [ConnectionState.STARTING, ConnectionState.FAILED]


def test_chat_state_payloads_are_enum_only() -> None:
    captured = []
    chat = _new_chat(state_change_handler=captured.append)
    chat._set_connection_state(ConnectionState.STARTING)
    chat._set_connection_state(ConnectionState.READY)
    assert captured == [ConnectionState.STARTING, ConnectionState.READY]
    assert all(isinstance(state, ConnectionState) for state in captured)
    text = repr(captured)
    for secret in ('wss://', 'sess-1', 'Bearer', 'Authorization', 'token'):
        assert secret not in text


def test_chat_state_handler_exception_is_swallowed() -> None:
    def handler(_state) -> None:
        raise RuntimeError('handler boom token=super-secret')

    chat = _new_chat(state_change_handler=handler)
    chat._set_connection_state(ConnectionState.STARTING)
    chat._set_connection_state(ConnectionState.READY)
    assert chat.connection_state is ConnectionState.READY


def test_chat_state_transitions_are_serialized() -> None:
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

    chat = _new_chat(state_change_handler=handler)
    first = threading.Thread(target=lambda: chat._set_connection_state(ConnectionState.READY))
    second = threading.Thread(target=lambda: chat._set_connection_state(ConnectionState.FAILED))
    try:
        first.start()
        assert entered.wait(1.0) is True
        assert chat.connection_state is ConnectionState.READY
        # The handler runs outside the state lock, so a concurrent transition can update the
        # stored state and be notified without waiting for the blocked READY handler.
        second.start()
        second.join(1.0)
        assert second.is_alive() is False
        assert chat.connection_state is ConnectionState.FAILED
        with calls_lock:
            assert calls == [ConnectionState.READY, ConnectionState.FAILED]
        # identical consecutive states must not re-notify
        chat._set_connection_state(ConnectionState.FAILED)
        with calls_lock:
            assert calls.count(ConnectionState.FAILED) == 1
    finally:
        release.set()
        first.join(1.0)
        second.join(1.0)
    assert chat.connection_state is ConnectionState.FAILED


def test_chat_state_handler_calling_stop_does_not_deadlock() -> None:
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
        # emulate __run_socket's finally surfacing the terminal state
        chat._set_connection_state(ConnectionState.STOPPED)

    socket_thread = threading.Thread(target=socket_body, daemon=True)

    def handler(state) -> None:
        calls.append(state)
        if state is ConnectionState.READY and not reentered.is_set():
            reentered.set()
            try:
                chat.stop(timeout=1.0)
            except BaseException as exc:  # noqa: BLE001 - recorded for the regression assertion
                errors.append(exc)

    chat = _new_chat(
        state_change_handler=handler,
        _Chat__running=True,
        _Chat__socket_loop=loop,
        _Chat__socket_thread=socket_thread,
    )
    chat._stop = fake_stop
    socket_thread.start()
    try:
        chat._set_connection_state(ConnectionState.READY)
        socket_thread.join(2.0)
    finally:
        socket_release.set()
        socket_thread.join(1.0)
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(1.0)
        loop.close()
    assert errors == []
    assert calls == [ConnectionState.READY, ConnectionState.STOPPING, ConnectionState.STOPPED]


# --- reconnect and failure projection (FIX 3) ---

async def _noop(*_args, **_kwargs) -> None:
    return None


async def _drain_cancelled(tasks) -> None:
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def test_chat_reconnect_reports_reconnecting() -> None:
    states = []
    chat = _new_chat(state_change_handler=states.append)
    chat._Chat__connect = _noop
    chat._Chat__task_startup = _noop
    asyncio.run(chat._handle_base_reconnect())
    assert states == [ConnectionState.RECONNECTING]


def test_chat_socket_exit_without_closing_reports_failed() -> None:
    states = []
    chat = _new_chat(state_change_handler=states.append, _closing=False)
    chat._Chat__connect = _noop
    chat._Chat__task_receive = _noop
    chat._Chat__task_startup = _noop
    chat._keep_loop_alive = _noop
    chat._Chat__run_socket()
    socket_loop = chat._Chat__socket_loop
    try:
        assert states == [ConnectionState.FAILED]
        assert chat.connection_state is ConnectionState.FAILED
    finally:
        socket_loop.run_until_complete(_drain_cancelled(chat._Chat__tasks))
        socket_loop.close()
        asyncio.set_event_loop(None)

