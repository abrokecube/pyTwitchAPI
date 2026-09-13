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
