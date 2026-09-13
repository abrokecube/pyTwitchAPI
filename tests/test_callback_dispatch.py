import asyncio
import logging
import threading
from collections import deque
from functools import partial

import pytest

from twitchAPI.chat import Chat
from twitchAPI.eventsub.websocket import EventSubWebsocket
from twitchAPI.helper import done_task_callback, submit_coroutine
from twitchAPI.type import ChatEvent


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
