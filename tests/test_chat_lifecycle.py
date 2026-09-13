import asyncio
import logging
import threading

import pytest

from twitchAPI.chat import Chat


class _FakeTwitch:
    def has_required_auth(self, *_args, **_kwargs) -> bool:
        return True


def _new_chat(**attrs) -> Chat:
    chat = object.__new__(Chat)
    chat.logger = logging.getLogger('test.chat')
    chat.twitch = _FakeTwitch()
    chat.username = 'testbot'
    chat._Chat__socket_thread = None
    chat._Chat__socket_loop = None
    chat._Chat__running = False
    chat._Chat__startup_complete = False
    chat._ready = False
    chat._closing = False
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
