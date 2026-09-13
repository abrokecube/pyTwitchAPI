import asyncio
import logging

import pytest

from twitchAPI.eventsub.websocket import EventSubWebsocket


class _FakeTwitch:
    session_timeout = 30

    def has_required_auth(self, *_args, **_kwargs) -> bool:
        return True


def _new_eventsub(**attrs) -> EventSubWebsocket:
    client = object.__new__(EventSubWebsocket)
    client.logger = logging.getLogger('test.eventsub.revocation')
    client._twitch = _FakeTwitch()
    client._callbacks = {}
    client._active_subscriptions = {}
    client._callback_loop = None
    client._task_callback = lambda _task: None
    client.revokation_handler = None
    for key, value in attrs.items():
        setattr(client, key, value)
    return client


async def _drain_tasks() -> None:
    await asyncio.sleep(0)
    await asyncio.sleep(0)


def _known_client(**attrs) -> EventSubWebsocket:
    return _new_eventsub(
        _callbacks={'sub-1': {'id': 'sub-1', 'callback': lambda _e: None, 'active': True, 'event': lambda **kw: kw}},
        _active_subscriptions={'sub-1': {'sub_type': 'channel.chat.message'}},
        _callback_loop=asyncio.get_running_loop(),
        **attrs,
    )


def _revocation(sub_id: str, status: str = 'user_removed') -> dict:
    return {
        'metadata': {'message_type': 'revocation'},
        'payload': {'subscription': {'id': sub_id, 'status': status}},
    }


def test_known_revocation_removes_state_and_dispatches_handler_once() -> None:
    normal_calls: list = []
    revoked: list = []

    async def event_callback(event) -> None:
        normal_calls.append(event)

    async def revocation_handler(payload) -> None:
        revoked.append(payload)

    async def scenario() -> None:
        client = _new_eventsub(
            _callbacks={'sub-1': {'id': 'sub-1', 'callback': event_callback, 'active': True, 'event': lambda **kw: kw}},
            _active_subscriptions={'sub-1': {'sub_type': 'channel.chat.message'}},
            _callback_loop=asyncio.get_running_loop(),
            revokation_handler=revocation_handler,
        )

        await client._handle_revocation(_revocation('sub-1'))
        await _drain_tasks()

        assert 'sub-1' not in client._active_subscriptions
        assert 'sub-1' not in client._callbacks
        assert len(revoked) == 1
        assert revoked[0]['subscription']['id'] == 'sub-1'
        assert normal_calls == []

    asyncio.run(scenario())


def test_known_revocation_without_handler_still_removes_state() -> None:
    async def scenario() -> None:
        client = _known_client()
        await client._handle_revocation(_revocation('sub-1'))
        await _drain_tasks()
        assert 'sub-1' not in client._active_subscriptions
        assert 'sub-1' not in client._callbacks

    asyncio.run(scenario())


def test_revoked_subscription_is_not_resubscribed() -> None:
    async def scenario() -> None:
        client = _new_eventsub(
            _callbacks={
                'sub-1': {'id': 'sub-1', 'callback': lambda _e: None, 'active': True, 'event': lambda **kw: kw},
                'sub-2': {'id': 'sub-2', 'callback': lambda _e: None, 'active': True, 'event': lambda **kw: kw},
            },
            _active_subscriptions={
                'sub-1': {'sub_type': 'channel.chat.message', 'sub_version': '1', 'condition': {}, 'callback': lambda _e: None, 'event': lambda **kw: kw},
                'sub-2': {'sub_type': 'stream.online', 'sub_version': '1', 'condition': {}, 'callback': lambda _e: None, 'event': lambda **kw: kw},
            },
            _callback_loop=asyncio.get_running_loop(),
        )
        await client._handle_revocation(_revocation('sub-1'))
        resubscribed: list = []

        async def fake_subscribe(sub_type, sub_version, condition, callback, event, is_batching_enabled=None):
            resubscribed.append(sub_type)
            return sub_type

        client._subscribe = fake_subscribe
        await client._resubscribe()

        assert resubscribed == ['stream.online']
        assert 'sub-1' not in resubscribed
        assert 'sub-1' not in client._active_subscriptions
        assert 'sub-1' not in client._callbacks

    asyncio.run(scenario())


def test_unknown_revocation_does_not_touch_unrelated_subscriptions() -> None:
    handler_calls: list = []

    async def revocation_handler(payload) -> None:
        handler_calls.append(payload)

    async def scenario() -> None:
        client = _known_client(revokation_handler=revocation_handler)
        await client._handle_revocation(_revocation('sub-unknown'))
        await _drain_tasks()
        assert 'sub-1' in client._active_subscriptions
        assert 'sub-1' in client._callbacks
        assert handler_calls == []

    asyncio.run(scenario())


_MALFORMED_REVOCATIONS = [
    {},
    {'payload': None},
    {'payload': 'not-a-dict'},
    {'payload': []},
    {'payload': {'subscription': None}},
    {'payload': {'subscription': 'not-a-dict'}},
    {'payload': {'subscription': []}},
    {'payload': {'subscription': {}}},
    {'payload': {'subscription': {'id': None}}},
    {'payload': {'subscription': {'id': ''}}},
    {'payload': {'subscription': {'id': 123}}},
    {'payload': {'subscription': {'id': []}}},
    {'payload': {'subscription': {'status': 'user_removed'}}},
]


@pytest.mark.parametrize('data', _MALFORMED_REVOCATIONS)
def test_malformed_revocation_is_ignored(data: dict) -> None:
    handler_calls: list = []

    async def revocation_handler(payload) -> None:
        handler_calls.append(payload)

    async def scenario() -> None:
        client = _known_client(revokation_handler=revocation_handler)
        await client._handle_revocation(data)
        await _drain_tasks()
        assert 'sub-1' in client._active_subscriptions
        assert 'sub-1' in client._callbacks
        assert handler_calls == []

    asyncio.run(scenario())
