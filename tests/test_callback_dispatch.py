import asyncio
import concurrent.futures
import logging
import threading
from collections import deque
from functools import partial

import pytest

from twitchAPI.chat import Chat
from twitchAPI.eventsub.websocket import EventSubWebsocket
from twitchAPI.helper import done_task_callback, notify_state_change, submit_coroutine
from twitchAPI.type import ChatEvent, ConnectionState


def _run_loop_on_thread() -> tuple:
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    return loop, thread


def _shutdown_loop(loop: asyncio.AbstractEventLoop, thread: threading.Thread) -> None:
    loop.call_soon_threadsafe(loop.stop)
    thread.join(1.0)
    loop.close()


def test_submit_coroutine_runs_on_target_loop_thread() -> None:
    loop, loop_thread = _run_loop_on_thread()
    observed = {}

    async def callback() -> str:
        observed['thread_id'] = threading.get_ident()
        observed['loop'] = asyncio.get_running_loop()
        return 'ok'

    try:
        future = submit_coroutine(loop, callback())
        assert future.result(timeout=1.0) == 'ok'
        assert observed['loop'] is loop
        assert observed['thread_id'] == loop_thread.ident
    finally:
        _shutdown_loop(loop, loop_thread)


def test_submit_coroutine_consumes_exceptions_and_keeps_loop_alive() -> None:
    loop, loop_thread = _run_loop_on_thread()
    logger = logging.getLogger('test.callback.dispatch')

    async def boom() -> None:
        raise RuntimeError('boom')

    async def ok() -> str:
        return 'still-alive'

    try:
        future = submit_coroutine(loop, boom(), on_done=partial(done_task_callback, logger))
        with pytest.raises(RuntimeError, match='boom'):
            future.result(timeout=1.0)
        assert future.exception() is not None
        surviving = submit_coroutine(loop, ok())
        assert surviving.result(timeout=1.0) == 'still-alive'
        assert loop_thread.is_alive() is True
    finally:
        _shutdown_loop(loop, loop_thread)


def test_submit_coroutine_closes_coroutine_when_scheduling_fails() -> None:
    loop = asyncio.new_event_loop()
    loop.close()

    async def callback() -> str:
        return 'ok'

    coroutine = callback()
    with pytest.raises(RuntimeError):
        submit_coroutine(loop, coroutine)
    assert coroutine.cr_frame is None


class _FakeTwitch:
    def has_required_auth(self, *_args, **_kwargs) -> bool:
        return True


def _new_chat(**attrs) -> Chat:
    chat = object.__new__(Chat)
    chat.logger = logging.getLogger('test.chat.dispatch')
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
    chat._command_handler = {}
    chat._command_middleware = []
    chat._command_specific_middleware = {}
    chat._callback_loop = None
    chat._configured_callback_loop = None
    chat._state_lock = threading.RLock()
    chat._task_callback = partial(done_task_callback, chat.logger)
    for key, value in attrs.items():
        setattr(chat, key, value)
    return chat


def _message() -> dict:
    chat = object.__new__(Chat)
    chat._prefix = '!'
    chat._channel_command_prefix = {}
    return chat._parse_irc_message('@tmi-sent-ts=1700000000000 :foo!foo@foo.tmi.twitch.tv PRIVMSG #bar :hello world')


def test_chat_event_callback_runs_on_configured_callback_loop() -> None:
    socket_loop, socket_thread = _run_loop_on_thread()
    callback_loop, callback_thread = _run_loop_on_thread()
    observed = {}
    completed = threading.Event()

    async def handler(message) -> None:
        observed['thread_id'] = threading.get_ident()
        observed['loop'] = asyncio.get_running_loop()
        completed.set()

    chat = _new_chat(_callback_loop=callback_loop, _event_handler={ChatEvent.MESSAGE: [handler]})
    parsed = _message()
    try:
        future = asyncio.run_coroutine_threadsafe(chat._handle_msg(parsed), socket_loop)
        future.result(2.0)
        assert completed.wait(2.0)
    finally:
        _shutdown_loop(socket_loop, socket_thread)
        _shutdown_loop(callback_loop, callback_thread)
    assert observed['loop'] is callback_loop
    assert observed['thread_id'] == callback_thread.ident


# --- callback loop re-derivation across restart (FIX 4) ---

async def _noop(*_args, **_kwargs) -> None:
    return None


async def _drain_cancelled(tasks) -> None:
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def test_chat_callback_loop_is_recomputed_on_restart() -> None:
    stale_loop = asyncio.new_event_loop()
    stale_loop.close()

    chat = _new_chat(_configured_callback_loop=None, _callback_loop=stale_loop)
    chat._Chat__connect = _noop
    chat._Chat__task_receive = _noop
    chat._Chat__task_startup = _noop
    chat._keep_loop_alive = _noop
    chat._Chat__run_socket()
    socket_loop = chat._Chat__socket_loop
    try:
        assert chat._callback_loop is socket_loop
        assert chat._callback_loop is not stale_loop
    finally:
        socket_loop.run_until_complete(_drain_cancelled(chat._Chat__tasks))
        socket_loop.close()
        asyncio.set_event_loop(None)


def test_eventsub_callback_loop_is_recomputed_on_restart() -> None:
    stale_loop = asyncio.new_event_loop()
    stale_loop.close()

    client = _new_eventsub(_configured_callback_loop=None, _callback_loop=stale_loop)
    client._connect = _noop
    client._task_receive = _noop
    client._task_reconnect_handler = _noop
    client._keep_loop_alive = _noop
    client._run_socket()
    socket_loop = client._socket_loop
    try:
        assert client._callback_loop is socket_loop
        assert client._callback_loop is not stale_loop
    finally:
        socket_loop.run_until_complete(_drain_cancelled(client._tasks))
        socket_loop.close()
        asyncio.set_event_loop(None)


def test_chat_user_supplied_callback_loop_is_preserved() -> None:
    user_loop = asyncio.new_event_loop()
    stale_loop = asyncio.new_event_loop()
    stale_loop.close()
    try:
        chat = _new_chat(_configured_callback_loop=user_loop, _callback_loop=stale_loop)
        chat._Chat__connect = _noop
        chat._Chat__task_receive = _noop
        chat._Chat__task_startup = _noop
        chat._keep_loop_alive = _noop
        chat._Chat__run_socket()
        socket_loop = chat._Chat__socket_loop
        try:
            assert chat._callback_loop is user_loop
        finally:
            socket_loop.run_until_complete(_drain_cancelled(chat._Chat__tasks))
            socket_loop.close()
            asyncio.set_event_loop(None)
    finally:
        user_loop.close()


# --- scheduling failures must not escape the socket loop (FIX 5) ---

def test_chat_dispatch_scheduling_failure_is_suppressed(caplog) -> None:
    closed_loop = asyncio.new_event_loop()
    closed_loop.close()

    async def scenario() -> None:
        chat = _new_chat(_callback_loop=closed_loop, _configured_callback_loop=closed_loop)

        async def callback() -> None:
            return None

        coroutine = callback()
        chat._dispatch_callback(coroutine)
        assert coroutine.cr_frame is None

    with caplog.at_level(logging.WARNING, logger='test.chat.dispatch'):
        asyncio.run(scenario())
    assert 'failed to schedule callback' in caplog.text
    assert 'Event loop is closed' not in caplog.text


def test_eventsub_dispatch_scheduling_failure_is_suppressed(caplog) -> None:
    closed_loop = asyncio.new_event_loop()
    closed_loop.close()

    async def scenario() -> None:
        client = _new_eventsub(_callback_loop=closed_loop, _configured_callback_loop=closed_loop)

        async def callback() -> None:
            return None

        coroutine = callback()
        client._dispatch_callback(coroutine)
        assert coroutine.cr_frame is None

    with caplog.at_level(logging.WARNING, logger='test.eventsub.dispatch'):
        asyncio.run(scenario())
    assert 'failed to schedule callback' in caplog.text
    assert 'Event loop is closed' not in caplog.text


# --- done callback and state handler safety (FIX 6) ---

def test_done_task_callback_ignores_cancelled_concurrent_future() -> None:
    logger = logging.getLogger('test.callback.dispatch')
    future = concurrent.futures.Future()
    assert future.cancel() is True
    done_task_callback(logger, future)


def test_done_task_callback_ignores_cancelled_asyncio_task() -> None:
    async def scenario() -> None:
        async def never() -> None:
            await asyncio.Event().wait()

        task = asyncio.ensure_future(never())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        done_task_callback(logging.getLogger('test.callback.dispatch'), task)

    asyncio.run(scenario())


def test_notify_state_change_does_not_swallow_base_exceptions() -> None:
    def handler(_state) -> None:
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        notify_state_change(handler, ConnectionState.READY, logging.getLogger('test.callback.dispatch'))


def test_chat_event_callback_defaults_to_socket_loop() -> None:
    async def scenario() -> None:
        observed = {}
        completed = asyncio.Event()

        async def handler(message) -> None:
            observed['loop'] = asyncio.get_running_loop()
            completed.set()

        loop = asyncio.get_running_loop()
        chat = _new_chat(_callback_loop=loop, _event_handler={ChatEvent.MESSAGE: [handler]})
        await chat._handle_msg(_message())
        await asyncio.wait_for(completed.wait(), 1.0)
        assert observed['loop'] is loop

    asyncio.run(scenario())


def _new_eventsub(**attrs) -> EventSubWebsocket:
    client = object.__new__(EventSubWebsocket)
    client.logger = logging.getLogger('test.eventsub.dispatch')
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
    client._task_callback = lambda _task: None
    client._reset_timeout = lambda: None
    for key, value in attrs.items():
        setattr(client, key, value)
    return client


def _notification(message_id: str) -> dict:
    return {
        'metadata': {'message_id': message_id, 'message_type': 'notification'},
        'payload': {'subscription': {'id': 'sub-1'}, 'event': {}},
    }


def test_eventsub_notification_callback_runs_on_configured_callback_loop() -> None:
    socket_loop, socket_thread = _run_loop_on_thread()
    callback_loop, callback_thread = _run_loop_on_thread()
    observed = {}
    completed = threading.Event()

    async def callback(event) -> None:
        observed['thread_id'] = threading.get_ident()
        observed['loop'] = asyncio.get_running_loop()
        completed.set()

    async def scenario() -> None:
        client = _new_eventsub(
            _callbacks={'sub-1': {'id': 'sub-1', 'callback': callback, 'active': True, 'event': lambda **kw: kw}},
            _msg_id_history=deque(maxlen=5),
            _callback_loop=callback_loop,
        )
        await client._handle_notification(_notification('m-cross-thread'))

    try:
        future = asyncio.run_coroutine_threadsafe(scenario(), socket_loop)
        future.result(2.0)
        assert completed.wait(2.0)
    finally:
        _shutdown_loop(socket_loop, socket_thread)
        _shutdown_loop(callback_loop, callback_thread)
    assert observed['loop'] is callback_loop
    assert observed['thread_id'] == callback_thread.ident


def test_eventsub_revocation_callback_runs_on_configured_callback_loop() -> None:
    socket_loop, socket_thread = _run_loop_on_thread()
    callback_loop, callback_thread = _run_loop_on_thread()
    observed = {}
    completed = threading.Event()

    async def revocation_handler(payload) -> None:
        observed['thread_id'] = threading.get_ident()
        observed['loop'] = asyncio.get_running_loop()
        completed.set()

    async def scenario() -> None:
        client = _new_eventsub(
            _callbacks={'sub-1': {'id': 'sub-1', 'callback': lambda _e: None, 'active': True, 'event': lambda **kw: kw}},
            _active_subscriptions={'sub-1': {}},
            _callback_loop=callback_loop,
            revokation_handler=revocation_handler,
        )
        await client._handle_revocation(
            {'payload': {'subscription': {'id': 'sub-1', 'status': 'user_removed'}}}
        )

    try:
        future = asyncio.run_coroutine_threadsafe(scenario(), socket_loop)
        future.result(2.0)
        assert completed.wait(2.0)
    finally:
        _shutdown_loop(socket_loop, socket_thread)
        _shutdown_loop(callback_loop, callback_thread)
    assert observed['loop'] is callback_loop
    assert observed['thread_id'] == callback_thread.ident
