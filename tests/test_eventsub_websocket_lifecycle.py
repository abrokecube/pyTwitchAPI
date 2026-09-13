import asyncio
import logging
import threading

import pytest

from twitchAPI.eventsub.websocket import EventSubWebsocket


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
    for key, value in attrs.items():
        setattr(client, key, value)
    return client


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
