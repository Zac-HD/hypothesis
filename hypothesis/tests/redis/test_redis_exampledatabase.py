# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import os
import uuid

import pytest
from fakeredis import FakeRedis
from redis import Redis

from hypothesis import settings, strategies as st
from hypothesis.database import InMemoryExampleDatabase, MapPut
from hypothesis.errors import InvalidArgument
from hypothesis.extra.redis import RedisExampleDatabase
from hypothesis.stateful import Bundle, RuleBasedStateMachine, rule

from tests.common.utils import time_sleep
from tests.cover.test_database_backend import _database_conforms_to_listener_api
from tests.cover.test_database_structured import (
    ForwardingDatabase,
    conforms_to_structured_api,
    journal_records_entries_deleted_by_id,
)

# Set this to a URL, such as redis://localhost:6379, to also test a real server.
REDIS_URL = os.environ.get("HYPOTHESIS_TEST_REDIS_URL")


@pytest.fixture(autouse=True)
def _consistently_increment_time():
    """Use the real clock, not the test suite's fake one, because these tests
    wait for other threads and processes."""


@pytest.mark.parametrize(
    "kw",
    [
        {"redis": "not a redis instance"},
        {"redis": FakeRedis(), "expire_after": 10},  # not a timedelta
        {"redis": FakeRedis(), "key_prefix": "not a bytestring"},
        {"redis": FakeRedis(), "listener_channel": 2},  # not a str
    ],
)
def test_invalid_args_raise(kw):
    with pytest.raises(InvalidArgument):
        RedisExampleDatabase(**kw)


def test_all_methods():
    db = RedisExampleDatabase(FakeRedis())
    db.save(b"key1", b"value")
    assert list(db.fetch(b"key1")) == [b"value"]
    db.move(b"key1", b"key2", b"value")
    assert list(db.fetch(b"key1")) == []
    assert list(db.fetch(b"key2")) == [b"value"]
    db.delete(b"key2", b"value")
    assert list(db.fetch(b"key2")) == []
    db.delete(b"key2", b"unknown value")


class DatabaseComparison(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        server = FakeRedis(host=uuid.uuid4().hex)  # Different (fake) server each time
        self.dbs = [InMemoryExampleDatabase(), RedisExampleDatabase(server)]

    keys = Bundle("keys")
    values = Bundle("values")

    @rule(target=keys, k=st.binary())
    def k(self, k):
        return k

    @rule(target=values, v=st.binary())
    def v(self, v):
        return v

    @rule(k=keys, v=values)
    def save(self, k, v):
        for db in self.dbs:
            db.save(k, v)

    @rule(k=keys, v=values)
    def delete(self, k, v):
        for db in self.dbs:
            db.delete(k, v)

    @rule(k1=keys, k2=keys, v=values)
    def move(self, k1, k2, v):
        for db in self.dbs:
            db.move(k1, k2, v)

    @rule(k=keys)
    def values_agree(self, k):
        last = None
        last_db = None
        for db in self.dbs:
            keys = set(db.fetch(k))
            if last is not None:
                assert last == keys, (last_db, db)
            last = keys
            last_db = db


TestDBs = DatabaseComparison.TestCase


def flush_messages(db):
    # A thread delivers messages to listeners, polling every 10ms.
    time_sleep(0.2)


def test_redis_listener():
    # A thread delivers the events, so steps wait for it, and can be slow.
    _database_conforms_to_listener_api(
        lambda _path: RedisExampleDatabase(FakeRedis()),
        flush=None,
        parent_settings=settings(max_examples=5, stateful_step_count=10, deadline=None),
    )


def test_redis_listener_explicit():
    calls = 0

    def listener(event):
        nonlocal calls
        calls += 1

    redis = FakeRedis()
    db = RedisExampleDatabase(redis)
    db.add_listener(listener)

    db.save(b"a", b"a")
    flush_messages(db)
    assert calls == 1

    db.remove_listener(listener)
    db.delete(b"a", b"a")
    db.save(b"a", b"b")
    flush_messages(db)
    assert calls == 1

    db.add_listener(listener)
    db.delete(b"a", b"b")
    db.save(b"a", b"c")
    flush_messages(db)
    assert calls == 3

    db.save(b"a", b"c")
    flush_messages(db)
    assert calls == 3
    db.close()


def test_redis_move_from_key_without_value():
    # explicit covering test for:
    # * moving a value from a key without that value
    redis = FakeRedis()
    db = RedisExampleDatabase(redis)
    db.save(b"a", b"x")
    db.save(b"b", b"x")
    db.move(b"a", b"b", b"y")


def test_redis_move_into_key_with_value():
    # explicit covering test for:
    # * moving a value into a key with that value
    redis = FakeRedis()
    db = RedisExampleDatabase(redis)
    db.save(b"a", b"y")
    db.save(b"b", b"x")
    db.move(b"a", b"b", b"x")


def test_redis_move_to_same_key():
    # explicit covering test for:
    # * moving a value where src == dest
    redis = FakeRedis()
    db = RedisExampleDatabase(redis)
    db.move(b"a", b"a", b"x")
    assert list(db.fetch(b"a")) == [b"x"]


def test_redis_equality():
    redis = FakeRedis()
    assert RedisExampleDatabase(redis) == RedisExampleDatabase(redis)
    # FakeRedis() != FakeRedis(), not much we can do here
    assert RedisExampleDatabase(FakeRedis()) != RedisExampleDatabase(FakeRedis())


def test_structured_api_fakeredis():
    conforms_to_structured_api(
        lambda _path: RedisExampleDatabase(FakeRedis(host=uuid.uuid4().hex)),
        parent_settings=settings(max_examples=20, stateful_step_count=25),
    )


@pytest.mark.parametrize("wrap", [lambda db: db, ForwardingDatabase])
def test_journal_records_entries_deleted_by_id_fakeredis(wrap):
    # The wrapper's journal comes from the firehose channel.
    db = RedisExampleDatabase(FakeRedis(host=uuid.uuid4().hex), listener_channel="x")
    with wrap(db) as db:
        journal_records_entries_deleted_by_id(db)


def scripts_are_reloaded(db):
    # After a restart, the server has no scripts. Nothing should be applied twice.
    db.map_put((b"p", "m"), "f", b"1")
    db.redis.script_flush()
    assert db.map_get((b"p", "m"), "f") == b"1"
    db.redis.script_flush()
    db.log_append((b"p", "log"), b"once")
    assert [value for _, value in db.log_range((b"p", "log"))] == [b"once"]
    db.redis.script_flush()
    batch = [MapPut((b"p", "m"), "f", b"2", expect=b"1")]
    assert db.write_many(batch, atomic=True) == [True]


def test_scripts_are_reloaded_fakeredis():
    scripts_are_reloaded(RedisExampleDatabase(FakeRedis(host=uuid.uuid4().hex)))


def _real_redis(_path):
    name = uuid.uuid4().hex
    return RedisExampleDatabase(
        Redis.from_url(REDIS_URL),
        key_prefix=f"test-{name}:".encode(),
        listener_channel=f"test-{name}",
    )


@pytest.mark.skipif(REDIS_URL is None, reason="needs HYPOTHESIS_TEST_REDIS_URL")
def test_structured_api_real_redis():
    conforms_to_structured_api(
        _real_redis, parent_settings=settings(max_examples=50, stateful_step_count=30)
    )


@pytest.mark.skipif(REDIS_URL is None, reason="needs HYPOTHESIS_TEST_REDIS_URL")
def test_scripts_are_reloaded_real_redis():
    scripts_are_reloaded(_real_redis(None))


@pytest.mark.skipif(REDIS_URL is None, reason="needs HYPOTHESIS_TEST_REDIS_URL")
def test_listener_api_real_redis():
    _database_conforms_to_listener_api(
        _real_redis,
        flush=None,
        parent_settings=settings(max_examples=5, stateful_step_count=10, deadline=None),
    )


@pytest.mark.skipif(REDIS_URL is None, reason="needs HYPOTHESIS_TEST_REDIS_URL")
def test_map_entries_expire_real_redis():
    db = _real_redis(None)
    db.map_put(("p", "m"), "short", b"1", ttl=0.5)
    db.map_put(("p", "m"), "long", b"2", ttl=100)
    time_sleep(1)
    assert db.map_items(("p", "m")) == {("long",): b"2"}
    db.map_delete(("p", "m"), "long")
    assert not db.redis.exists(db._data_keys(("p", "m"))[0])
