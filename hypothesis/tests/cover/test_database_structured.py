# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Conformance tests for maps, logs, batches, and the journal.

``conforms_to_structured_api`` runs one state machine against a database and a
plain Python model. Other test modules use it for the Redis and Postgres backends.
"""

import pickle
import shutil
import tempfile
import threading
import time
from collections import defaultdict
from pathlib import Path

import pytest

from hypothesis import HealthCheck, settings, strategies as st
from hypothesis.database import (
    BackgroundWriteDatabase,
    DirectoryBasedExampleDatabase,
    ExampleDatabase,
    InMemoryExampleDatabase,
    JournalCursorExpired,
    MapDelete,
    MapPut,
    MultiplexedDatabase,
    ReadOnlyDatabase,
    ReadThroughDatabase,
    RemoteDatabase,
    SQLiteExampleDatabase,
    serve_database,
    unset,
)
from hypothesis.errors import InvalidArgument
from hypothesis.internal.dbcodec import is_legacy, make_cursor
from hypothesis.stateful import (
    RuleBasedStateMachine,
    precondition,
    rule,
    run_state_machine_as_test,
)

from tests.common.utils import wait_for
from tests.cover.test_database_backend import _database_conforms_to_listener_api

LEGACY_KEYS = [(b"a",), (b"b",)]
STRUCTURED_KEYS = [(b"a", "m"), (b"a", "n", 1), (b"b", "m"), ("index", "x")]
MAP_KEYS = LEGACY_KEYS + STRUCTURED_KEYS
LOG_KEYS = [(b"a", "log"), ("index", "log")]
PARTITIONS = [b"a", b"b", "index"]

legacy_fields = st.tuples(st.binary(max_size=2))
structured_fields = st.one_of(
    st.just(()),
    st.tuples(st.binary(max_size=2)),
    st.tuples(st.sampled_from(["x", "y"]), st.integers(-2, 2)),
)
values = st.binary(max_size=4)


class ForwardingDatabase(ExampleDatabase):
    """Implements only the old methods, so everything else is emulated."""

    def __init__(self, db):
        super().__init__()
        self._db = db

    def save(self, key, value):
        self._db.save(key, value)

    def fetch(self, key):
        return self._db.fetch(key)

    def delete(self, key, value):
        self._db.delete(key, value)

    def _start_listening(self):
        self._db.add_listener(self._broadcast_change)

    def _stop_listening(self):
        self._db.remove_listener(self._broadcast_change)


def _is_suffix(short, long):
    return len(short) <= len(long) and long[len(long) - len(short) :] == short


def conforms_to_structured_api(create_db, *, journal=True, parent_settings=None):
    """Check a database against a model.

    ``create_db`` takes a temporary directory and returns a database. If the
    database has a ``_test_cleanup`` attribute, it is called at the end. With
    ``journal=True``, a mirror built from the journal must match the model.
    """

    @settings(
        parent_settings,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    class StructuredMachine(RuleBasedStateMachine):
        def __init__(self):
            super().__init__()
            self.tmp = Path(tempfile.mkdtemp())
            self.db = create_db(self.tmp)
            self.maps = defaultdict(dict)
            self.logs = defaultdict(list)
            self.mirror_maps = defaultdict(dict)
            self.mirror_logs = defaultdict(list)
            self.journal = journal
            self.cursors = (
                {p: self.db.journal_head(p) for p in PARTITIONS} if journal else {}
            )

        def teardown(self):
            getattr(self.db, "_test_cleanup", lambda: None)()
            shutil.rmtree(self.tmp, ignore_errors=True)

        def _field(self, data, key, *, existing=False):
            fields = legacy_fields if is_legacy(key) else structured_fields
            if existing and self.maps[key]:
                fields = st.sampled_from(sorted(self.maps[key], key=repr)) | fields
            return data.draw(fields)

        @rule(data=st.data(), key=st.sampled_from(MAP_KEYS))
        def put(self, data, key):
            field = self._field(data, key, existing=True)
            value = b"" if is_legacy(key) else data.draw(values)
            current = self.maps[key].get(field)
            expect = data.draw(st.sampled_from([unset, None, current, b"zz"]))
            applies = expect is unset or current == expect
            result = self.db.map_put(key, field, value, expect=expect)
            assert result == applies or (result is None and expect is unset)
            if applies:
                self.maps[key][field] = value

        @rule(data=st.data(), key=st.sampled_from(MAP_KEYS))
        def delete(self, data, key):
            field = self._field(data, key, existing=True)
            current = self.maps[key].get(field)
            expect = data.draw(st.sampled_from([unset, current or b"", b"zz"]))
            applies = current is not None and (expect is unset or current == expect)
            result = self.db.map_delete(key, field, expect=expect)
            assert result == applies or (result is None and expect is unset)
            if applies:
                del self.maps[key][field]

        @rule(key=st.sampled_from(MAP_KEYS))
        def clear(self, key):
            self.db.map_clear(key)
            self.maps[key].clear()

        @rule(data=st.data(), key=st.sampled_from(MAP_KEYS))
        def get(self, data, key):
            field = self._field(data, key, existing=True)
            assert self.db.map_get(key, field) == self.maps[key].get(field)

        @rule(data=st.data(), key=st.sampled_from(MAP_KEYS))
        def items(self, data, key):
            prefixes = {f[:n] for f in self.maps[key] for n in range(len(f) + 1)}
            prefix = data.draw(st.sampled_from(sorted(prefixes | {()}, key=repr)))
            expected = {
                f: v for f, v in self.maps[key].items() if f[: len(prefix)] == prefix
            }
            assert self.db.map_items(key, prefix=prefix) == expected

        @rule(
            key=st.sampled_from(LOG_KEYS),
            value=values,
            maxlen=st.none() | st.integers(1, 4),
        )
        def append(self, key, value, maxlen):
            result = self.db.log_append(key, value, maxlen=maxlen)
            assert result is None or len(result) == 16
            expected = [*self.logs[key], value]
            entries = self.db.log_range(key)
            actual = [v for _, v in entries]
            keep = len(expected) if maxlen is None else min(maxlen, len(expected))
            assert len(actual) >= keep
            assert _is_suffix(actual, expected)
            ids = [i for i, _ in entries]
            assert ids == sorted(set(ids))
            self.logs[key] = actual

        @rule(data=st.data(), key=st.sampled_from(LOG_KEYS))
        def log_range(self, data, key):
            entries = self.db.log_range(key)
            assert [v for _, v in entries] == self.logs[key]
            ids = [i for i, _ in entries] or [bytes(16)]
            after = data.draw(st.none() | st.sampled_from(ids))
            before = data.draw(st.none() | st.sampled_from(ids))
            limit = data.draw(st.none() | st.integers(0, 3))
            reverse = data.draw(st.booleans())
            expected = [
                e
                for e in entries
                if (after is None or e[0] > after) and (before is None or e[0] < before)
            ]
            if reverse:
                expected.reverse()
            if limit is not None:
                expected = expected[:limit]
            actual = self.db.log_range(
                key, after=after, before=before, limit=limit, reverse=reverse
            )
            assert actual == expected

        @rule(data=st.data(), key=st.sampled_from(LOG_KEYS))
        def trim(self, data, key):
            entries = self.db.log_range(key)
            maxlen = data.draw(st.none() | st.integers(0, 4))
            before = data.draw(
                st.none() | st.sampled_from([i for i, _ in entries] or [bytes(16)])
            )
            keep = (
                entries
                if maxlen is None
                else entries[len(entries) - min(maxlen, len(entries)) :]
            )
            keep = [e for e in keep if before is None or e[0] >= before]
            removed = self.db.log_trim(key, maxlen=maxlen, before=before)
            assert removed in (len(entries) - len(keep), None)
            self.logs[key] = [v for _, v in keep]

        def _draw_ops(self, data, keys):
            ops = []
            for key in data.draw(
                st.lists(st.sampled_from(keys), min_size=1, max_size=3)
            ):
                field = self._field(data, key, existing=True)
                current = self.maps[key].get(field)
                if data.draw(st.booleans()):
                    value = b"" if is_legacy(key) else data.draw(values)
                    expect = data.draw(
                        st.sampled_from([unset, unset, None, current, b"zz"])
                    )
                    ops.append(MapPut(key, field, value, expect=expect))
                else:
                    expect = data.draw(
                        st.sampled_from([unset, unset, current or b"", b"zz"])
                    )
                    ops.append(MapDelete(key, field, expect=expect))
            return ops

        def _apply(self, op, *, check):
            current = self.maps[op.key].get(op.field)
            if check and op.expect is not unset and current != op.expect:
                return False
            if isinstance(op, MapPut):
                self.maps[op.key][op.field] = op.value
                return True
            self.maps[op.key].pop(op.field, None)
            return current is not None

        @rule(data=st.data(), partition=st.sampled_from(PARTITIONS))
        def atomic_batch(self, data, partition):
            ops = self._draw_ops(data, [k for k in MAP_KEYS if k[0] == partition])
            holds = all(
                op.expect is unset or self.maps[op.key].get(op.field) == op.expect
                for op in ops
            )
            expected = (
                [self._apply(op, check=False) for op in ops]
                if holds
                else [False] * len(ops)
            )
            assert self.db.write_many(ops, atomic=True) == expected

        @rule(data=st.data())
        def batch(self, data):
            ops = self._draw_ops(data, MAP_KEYS)
            conditional = any(op.expect is not unset for op in ops)
            expected = [self._apply(op, check=True) for op in ops]
            result = self.db.write_many(ops)
            assert result == expected or (
                not conditional and result == [None] * len(ops)
            )

        def _apply_change(self, change):
            if change.op == "append":
                self.mirror_logs[change.key].append(change.value)
                return
            fields = self.mirror_maps[change.key]
            if change.op == "clear" or (
                change.op == "invalidate" and change.field is None
            ):
                fields.clear()
                if change.op == "invalidate":
                    fields.update(self.db.map_items(change.key))
            elif change.op == "delete":
                fields.pop(change.field, None)
            else:
                value = change.value
                if change.op == "invalidate" or value is None:
                    value = self.db.map_get(change.key, change.field)
                if value is None:
                    fields.pop(change.field, None)
                else:
                    fields[change.field] = value

        def _mirror_matches(self):
            while True:
                changes, self.cursors = self.db.journal_read(self.cursors, timeout=0)
                for change in changes:
                    self._apply_change(change)
                if not changes:
                    break
            return all(self.mirror_maps[k] == self.maps[k] for k in MAP_KEYS) and all(
                _is_suffix(self.logs[k], self.mirror_logs[k]) for k in LOG_KEYS
            )

        @precondition(lambda self: self.journal)
        @rule()
        def check_journal(self):
            self.db.flush()
            try:
                wait_for(self._mirror_matches, timeout=20)
            except Exception:
                for key in MAP_KEYS:
                    assert self.mirror_maps[key] == self.maps[key], key
                for key in LOG_KEYS:
                    assert self.mirror_logs[key] == self.logs[key], key
                raise

    run_state_machine_as_test(StructuredMachine)


def _served(create):
    def create_db(tmp):
        server = serve_database(create(tmp))
        client = server.client()
        client._test_cleanup = lambda: (client.close(), server.close())
        return client

    return create_db


def _forwarding_sqlite(tmp):
    inner = SQLiteExampleDatabase(tmp / "db.sqlite", poll_interval=0.001)
    db = ForwardingDatabase(inner)
    # Stops the listener thread, which the emulated journal starts.
    db._test_cleanup = inner.clear_listeners
    return db


BACKENDS = {
    "memory": lambda tmp: InMemoryExampleDatabase(),
    "sqlite": lambda tmp: SQLiteExampleDatabase(tmp / "db.sqlite"),
    "forwarding-memory": lambda tmp: ForwardingDatabase(InMemoryExampleDatabase()),
    "forwarding-sqlite": lambda tmp: _forwarding_sqlite(tmp),
    "background-memory": lambda tmp: BackgroundWriteDatabase(InMemoryExampleDatabase()),
    "multiplexed-memory": lambda tmp: MultiplexedDatabase(InMemoryExampleDatabase()),
    "read-through-memory": lambda tmp: ReadThroughDatabase(
        InMemoryExampleDatabase(), InMemoryExampleDatabase()
    ),
    "remote-memory": _served(lambda tmp: InMemoryExampleDatabase()),
    "remote-sqlite": _served(lambda tmp: SQLiteExampleDatabase(tmp / "db.sqlite")),
}


@pytest.mark.parametrize("name", sorted(BACKENDS))
def test_structured_api(name):
    conforms_to_structured_api(
        BACKENDS[name],
        parent_settings=settings(max_examples=20, stateful_step_count=25),
    )


def test_structured_api_directory():
    # The directory database's journal comes from watchdog, whose tests are
    # skipped as flaky, so check everything except the journal here.
    conforms_to_structured_api(
        lambda tmp: DirectoryBasedExampleDatabase(tmp / "dir"),
        journal=False,
        parent_settings=settings(max_examples=10, stateful_step_count=20),
    )


def test_sqlite_listener_api(tmp_path):
    _database_conforms_to_listener_api(
        lambda path: SQLiteExampleDatabase(path / "db.sqlite", poll_interval=0.001),
        flush=lambda db: db._wait_for_listeners(),
        parent_settings=settings(max_examples=5, stateful_step_count=10),
    )


@pytest.mark.parametrize("name", ["memory", "sqlite"])
def test_map_entries_expire(name, tmp_path):
    db = BACKENDS[name](tmp_path)
    db.map_put(("p", "m"), "short", b"1", ttl=1)
    db.map_put(("p", "m"), "long", b"2", ttl=100)
    db.map_put(("p", "m"), "forever", b"3", ttl=1)
    db.map_put(("p", "m"), "forever", b"3")
    time.sleep(2)  # advances the test suite's clock, which both backends use
    assert db.map_items(("p", "m")) == {("long",): b"2", ("forever",): b"3"}
    assert db.map_put(("p", "m"), "short", b"4", expect=None)


@pytest.mark.parametrize("name", sorted(BACKENDS))
def test_atomic_batches_must_use_one_partition(name, tmp_path):
    db = BACKENDS[name](tmp_path)
    with pytest.raises(InvalidArgument):
        db.write_many([MapPut(("p", "m"), "f"), MapPut(("q", "m"), "f")], atomic=True)
    getattr(db, "_test_cleanup", lambda: None)()


@pytest.mark.parametrize(
    "call",
    [
        lambda db: db.map_put(b"set", b"member", b"a value"),
        lambda db: db.map_put(b"set", ("not", "bytes")),
        lambda db: db.log_append(b"set", b"value"),
        lambda db: db.map_put((), b"field"),
        lambda db: db.map_put(("p", 1.5), b"field"),
        lambda db: db.log_append(("p", "log"), b"v", maxlen=0),
        lambda db: db.map_put(("p", "m"), "f", ttl=-1),
    ],
)
def test_invalid_arguments(call):
    with pytest.raises(InvalidArgument):
        call(InMemoryExampleDatabase())


@pytest.mark.parametrize("name", ["memory", "sqlite", "remote-memory"])
def test_old_cursors_expire(name, tmp_path):
    db = BACKENDS[name](tmp_path)
    cursor = db.journal_head("p")
    _, position = cursor[:8], cursor[8:]
    stale = make_cursor(position, issued_at=time.time() - db.journal_retention)
    with pytest.raises(JournalCursorExpired) as info:
        db.journal_read({"p": stale})
    assert info.value.partitions == ["p"]
    assert pickle.loads(pickle.dumps(info.value)).partitions == ["p"]
    getattr(db, "_test_cleanup", lambda: None)()


@pytest.mark.parametrize("name", ["memory", "remote-memory"])
def test_journal_read_wakes_up_for_a_change(name, tmp_path):
    db = BACKENDS[name](tmp_path)
    cursors = {"p": db.journal_head("p")}
    timer = threading.Timer(0.2, lambda: db.map_put(("p", "m"), "f", b"v"))
    timer.start()
    changes, _ = db.journal_read(cursors, timeout=30)
    timer.join()
    assert [(c.op, c.key, c.field, c.value) for c in changes] == [
        ("put", ("p", "m"), ("f",), b"v")
    ]
    getattr(db, "_test_cleanup", lambda: None)()


def test_only_the_journal_of_followed_partitions_is_read():
    db = InMemoryExampleDatabase()
    cursors = {"p": db.journal_head("p")}
    db.map_put(("q", "m"), "f", b"ignored")
    db.map_put(("p", "m"), "f", b"seen")
    changes, _ = db.journal_read(cursors)
    assert [c.partition for c in changes] == ["p"]


def test_old_style_databases_see_structured_data_written_natively(tmp_path):
    native = SQLiteExampleDatabase(tmp_path / "db.sqlite")
    native.map_put((b"T", "corpus"), b"choices", b"observation")
    native.log_append((b"T", "log"), b"entry")
    forwarding = ForwardingDatabase(native)
    assert forwarding.map_items((b"T", "corpus")) == {(b"choices",): b"observation"}
    assert [v for _, v in forwarding.log_range((b"T", "log"))] == [b"entry"]
    forwarding.map_put((b"T", "corpus"), b"more", b"data")
    assert native.map_get((b"T", "corpus"), b"more") == b"data"


def test_read_through_copies_each_key_once():
    primary, fallback = InMemoryExampleDatabase(), InMemoryExampleDatabase()
    fallback.save(b"key", b"old")
    db = ReadThroughDatabase(primary, fallback)
    assert set(db.fetch(b"key")) == {b"old"}
    db.delete(b"key", b"old")
    fallback.save(b"key", b"added later")
    assert set(ReadThroughDatabase(primary, fallback).fetch(b"key")) == set()


def test_read_only_database_ignores_structured_writes():
    inner = InMemoryExampleDatabase()
    db = ReadOnlyDatabase(inner)
    assert db.map_put(("p", "m"), "f", b"v") is False
    assert db.log_append(("p", "log"), b"v") is None
    assert inner.map_items(("p", "m")) == {}


def test_remote_database_can_be_pickled(tmp_path):
    with serve_database(InMemoryExampleDatabase()) as server:
        db = pickle.loads(pickle.dumps(server.client()))
        assert isinstance(db, RemoteDatabase)
        db.save(b"k", b"v")
        assert list(db.fetch(b"k")) == [b"v"]
        assert "native" in db.capabilities
