# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import abc
import enum
import errno
import json
import os
import random
import sqlite3
import struct
import sys
import tempfile
import threading
import time
import warnings
import weakref
from bisect import bisect_left, bisect_right
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from hashlib import sha384
from multiprocessing.connection import Client
from os import PathLike, getenv
from pathlib import Path, PurePath
from queue import Queue
from threading import Thread
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    NamedTuple,
    TypeAlias,
    TypeGuard,
    TypeVar,
    cast,
)
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

from hypothesis.configuration import StorageDirectory, storage_directory
from hypothesis.errors import HypothesisException, HypothesisWarning, InvalidArgument
from hypothesis.internal.conjecture.choice import ChoiceT
from hypothesis.internal.dbcodec import (
    ENTRY_ID_SIZE,
    KeyPartT,
    KeyTupleT,
    _pack_uleb128,
    _unpack_uleb128,
    as_tuple,
    decode,
    encode,
    field_key,
    index_key,
    is_legacy,
    log_key,
    make_cursor,
    make_entry_id,
    next_entry_id,
    parse_legacy_key,
    short_hash,
    split_cursor,
    split_entry_id,
)
from hypothesis.utils.conventions import UniqueIdentifier, not_set
from hypothesis.utils.deprecation import note_deprecation

__all__ = [
    "BackgroundWriteDatabase",
    "Change",
    "DirectoryBasedExampleDatabase",
    "ExampleDatabase",
    "GitHubArtifactDatabase",
    "InMemoryExampleDatabase",
    "JournalCursorExpired",
    "LogAppend",
    "LogRange",
    "LogTrim",
    "MapClear",
    "MapDelete",
    "MapGet",
    "MapItems",
    "MapPut",
    "MultiplexedDatabase",
    "ReadOnlyDatabase",
    "ReadThroughDatabase",
    "RemoteDatabase",
    "SQLiteExampleDatabase",
    "serve_database",
]

if TYPE_CHECKING:
    from watchdog.observers.api import BaseObserver

    from hypothesis.internal.dbserver import DatabaseServer

StrPathT: TypeAlias = str | PathLike[str]
SaveDataT: TypeAlias = tuple[bytes, bytes]  # key, value
DeleteDataT: TypeAlias = tuple[bytes, bytes | None]  # key, value
ListenerEventT: TypeAlias = (
    tuple[Literal["save"], SaveDataT] | tuple[Literal["delete"], DeleteDataT]
)
ListenerT: TypeAlias = Callable[[ListenerEventT], Any]
KeyT: TypeAlias = KeyPartT | KeyTupleT
_DatabaseT = TypeVar("_DatabaseT", bound="ExampleDatabase")
TTLT: TypeAlias = float | timedelta | None


class _Unset(enum.Enum):
    unset = "unset"

    def __repr__(self) -> str:
        return "unset"


#: The default for ``expect=``, meaning that the write is unconditional.
unset = _Unset.unset


def _normalize_key(key: object) -> KeyTupleT:
    return as_tuple(key, what="key", allow_empty=False)


def _normalize_field(key: KeyTupleT, field: object) -> KeyTupleT:
    normalized = as_tuple(field, what="field", allow_empty=True)
    if is_legacy(key) and not (
        len(normalized) == 1 and isinstance(normalized[0], bytes)
    ):
        raise InvalidArgument(
            f"The key {key!r} holds a set, so each field must be a single bytes "
            f"component, not {field!r}"
        )
    return normalized


def _normalize_ttl(ttl: TTLT) -> float | None:
    if ttl is None:
        return None
    seconds = ttl.total_seconds() if isinstance(ttl, timedelta) else float(ttl)
    if not seconds > 0:
        raise InvalidArgument(f"ttl must be positive, not {ttl!r}")
    return seconds


def _check_bytes(value: object, name: str) -> bytes:
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    if not isinstance(value, bytes):
        raise InvalidArgument(f"{name} must be bytes, not {value!r}")
    return value


def _check_count(value: int | None, name: str, *, minimum: int) -> None:
    if value is not None and (not isinstance(value, int) or value < minimum):
        raise InvalidArgument(f"{name} must be an integer >= {minimum}, not {value!r}")


def _check_entry_ids(*ids: bytes | None) -> None:
    for entry_id in ids:
        if entry_id is not None:
            split_entry_id(entry_id)


def _check_log_key(key: KeyTupleT) -> None:
    if is_legacy(key):
        raise InvalidArgument(f"The key {key!r} holds a set, so it cannot hold a log")


class _Op:
    __slots__ = ()

    def _init(self, **fields: object) -> None:
        for name, value in fields.items():
            object.__setattr__(self, name, value)


# Operations accept keys and fields in any form, and store them as tuples.


@dataclass(frozen=True, slots=True, init=False)
class MapGet(_Op):
    """Read one field of a map. The result is the value, or ``None``."""

    key: KeyTupleT
    field: KeyTupleT

    def __init__(self, key: KeyT, field: KeyT) -> None:
        normalized = _normalize_key(key)
        self._init(key=normalized, field=_normalize_field(normalized, field))


@dataclass(frozen=True, slots=True, init=False)
class MapItems(_Op):
    """Read every field of a map that extends ``prefix``, as a dict."""

    key: KeyTupleT
    prefix: KeyTupleT

    def __init__(self, key: KeyT, *, prefix: KeyT = ()) -> None:
        self._init(
            key=_normalize_key(key),
            prefix=as_tuple(prefix, what="prefix", allow_empty=True),
        )


@dataclass(frozen=True, slots=True, init=False)
class LogRange(_Op):
    """Read log entries as ``(entry_id, value)`` pairs, oldest first unless ``reverse``.

    ``after`` and ``before`` are exclusive bounds on the entry id.
    """

    key: KeyTupleT
    after: bytes | None
    before: bytes | None
    limit: int | None
    reverse: bool

    def __init__(
        self,
        key: KeyT,
        *,
        after: bytes | None = None,
        before: bytes | None = None,
        limit: int | None = None,
        reverse: bool = False,
    ) -> None:
        normalized = _normalize_key(key)
        _check_log_key(normalized)
        _check_count(limit, "limit", minimum=0)
        _check_entry_ids(after, before)
        self._init(
            key=normalized, after=after, before=before, limit=limit, reverse=reverse
        )


@dataclass(frozen=True, slots=True, init=False)
class MapPut(_Op):
    """Set a field of a map. See :meth:`ExampleDatabase.map_put`."""

    key: KeyTupleT
    field: KeyTupleT
    value: bytes
    ttl: float | None
    expect: bytes | _Unset | None

    def __init__(
        self,
        key: KeyT,
        field: KeyT,
        value: bytes = b"",
        *,
        ttl: TTLT = None,
        expect: bytes | _Unset | None = unset,
    ) -> None:
        normalized = _normalize_key(key)
        value = _check_bytes(value, "value")
        if is_legacy(normalized) and value:
            raise InvalidArgument(
                f"The key {normalized!r} holds a set, so values must be empty"
            )
        self._init(
            key=normalized,
            field=_normalize_field(normalized, field),
            value=value,
            ttl=_normalize_ttl(ttl),
            expect=(
                expect
                if expect is unset or expect is None
                else _check_bytes(expect, "expect")
            ),
        )


@dataclass(frozen=True, slots=True, init=False)
class MapDelete(_Op):
    """Delete a field of a map, if it exists and matches ``expect``."""

    key: KeyTupleT
    field: KeyTupleT
    expect: bytes | _Unset

    def __init__(
        self, key: KeyT, field: KeyT, *, expect: bytes | _Unset = unset
    ) -> None:
        normalized = _normalize_key(key)
        self._init(
            key=normalized,
            field=_normalize_field(normalized, field),
            expect=expect if expect is unset else _check_bytes(expect, "expect"),
        )


@dataclass(frozen=True, slots=True, init=False)
class MapClear(_Op):
    """Delete every field of a map."""

    key: KeyTupleT

    def __init__(self, key: KeyT) -> None:
        self._init(key=_normalize_key(key))


@dataclass(frozen=True, slots=True, init=False)
class LogAppend(_Op):
    """Append an entry to a log. See :meth:`ExampleDatabase.log_append`."""

    key: KeyTupleT
    value: bytes
    maxlen: int | None
    ttl: float | None

    def __init__(
        self, key: KeyT, value: bytes, *, maxlen: int | None = None, ttl: TTLT = None
    ) -> None:
        normalized = _normalize_key(key)
        _check_log_key(normalized)
        _check_count(maxlen, "maxlen", minimum=1)
        self._init(
            key=normalized,
            value=_check_bytes(value, "value"),
            maxlen=maxlen,
            ttl=_normalize_ttl(ttl),
        )


@dataclass(frozen=True, slots=True, init=False)
class LogTrim(_Op):
    """Remove the entries whose ids are in ``ids``, and the entries before
    ``before``, and then all but the newest ``maxlen`` entries."""

    key: KeyTupleT
    maxlen: int | None
    before: bytes | None
    ids: tuple[bytes, ...]

    def __init__(
        self,
        key: KeyT,
        *,
        maxlen: int | None = None,
        before: bytes | None = None,
        ids: Iterable[bytes] = (),
    ) -> None:
        normalized = _normalize_key(key)
        _check_log_key(normalized)
        _check_count(maxlen, "maxlen", minimum=0)
        ids = tuple(ids)
        _check_entry_ids(before, *ids)
        self._init(key=normalized, maxlen=maxlen, before=before, ids=ids)


ReadOpT: TypeAlias = MapGet | MapItems | LogRange
WriteOpT: TypeAlias = MapPut | MapDelete | MapClear | LogAppend | LogTrim
ChangeOpT: TypeAlias = Literal["put", "delete", "clear", "append", "invalidate"]


@dataclass(frozen=True, slots=True)
class Change:
    """One entry in a partition's journal. See :meth:`ExampleDatabase.journal_read`."""

    op: ChangeOpT
    key: KeyTupleT
    field: KeyTupleT | None = None
    entry_id: bytes | None = None
    value: bytes | None = None

    @property
    def partition(self) -> KeyPartT:
        return self.key[0]


class JournalCursorExpired(HypothesisException):
    """Raised by :meth:`ExampleDatabase.journal_read` when changes may have been lost.

    Reload the data for each of ``partitions``, and continue from a new cursor
    taken with :meth:`ExampleDatabase.journal_head` before reloading.
    """

    def __init__(self, partitions: Iterable[KeyPartT]) -> None:
        self.partitions = list(partitions)
        super().__init__(
            f"Journal cursors expired for partitions {self.partitions!r}. Reload "
            "them, then continue from journal_head()."
        )

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (self.partitions,))


def _is_conditional(op: object) -> TypeGuard[MapPut | MapDelete]:
    return isinstance(op, (MapPut, MapDelete)) and op.expect is not unset


def _check_atomic(ops: Sequence[WriteOpT]) -> None:
    partitions = {op.key[0] for op in ops}
    if len(partitions) > 1:
        raise InvalidArgument(
            "An atomic batch must write to a single partition, but this one writes "
            f"to {sorted(map(repr, partitions))}"
        )


def _not_applied(ops: Sequence[WriteOpT]) -> list[Any]:
    return [
        (
            False
            if isinstance(op, (MapPut, MapDelete))
            else 0 if isinstance(op, LogTrim) else None
        )
        for op in ops
    ]


def _matches_prefix(field: KeyTupleT, prefix: KeyTupleT) -> bool:
    return field[: len(prefix)] == prefix


def _select_entries(
    entries: list[tuple[bytes, bytes]], op: LogRange
) -> list[tuple[bytes, bytes]]:
    """Apply the bounds, direction, and limit of ``op`` to entries sorted by id."""
    lo, hi = 0, len(entries)
    if op.after is not None:
        lo = bisect_right(entries, op.after, key=lambda e: e[0])
    if op.before is not None:
        hi = bisect_left(entries, op.before, key=lambda e: e[0])
    selected = entries[lo:hi]
    if op.reverse:
        selected.reverse()
    return selected if op.limit is None else selected[: op.limit]


def _change_from_event(event: ListenerEventT) -> Change | None:
    """Translate an old-style listener event into a journal entry."""
    kind, (raw, value) = event
    parsed = parse_legacy_key(raw)
    if parsed.kind == "set":
        if value is None:
            return Change("invalidate", parsed.key)
        if kind == "save":
            return Change("put", parsed.key, (value,), value=b"")
        return Change("delete", parsed.key, (value,))
    if parsed.kind == "field":
        if kind == "save":
            return Change("put", parsed.key, parsed.field, value=value)
        # The value in a field key can be replaced by saving the new value and
        # then deleting the old one, so a deletion only tells us to re-read.
        return Change("invalidate", parsed.key, parsed.field)
    if parsed.kind == "index":
        # The fields changed, as when another client cleared the map.
        return None if kind == "save" else Change("invalidate", parsed.key)
    if parsed.kind == "log":
        if value is None or len(value) < ENTRY_ID_SIZE:
            # Something in the log changed, but the event does not say what.
            return None if kind == "save" else Change("invalidate", parsed.key)
        entry_id, rest = value[:ENTRY_ID_SIZE], value[ENTRY_ID_SIZE:]
        op: ChangeOpT = "append" if kind == "save" else "delete"
        return Change(op, parsed.key, entry_id=entry_id, value=rest)
    return None


def _events_from_change(change: Change) -> list[ListenerEventT]:
    """Translate a journal entry into old-style listener events."""
    if is_legacy(change.key):
        raw = cast(bytes, change.key[0])
        member = None if change.field is None else cast(bytes, change.field[0])
        if change.op == "put" and member is not None:
            return [("save", (raw, member))]
        return [("delete", (raw, member if change.op == "delete" else None))]
    ek = encode(change.key)
    if change.entry_id is not None:  # an entry of a log was appended or deleted
        if change.value is None:  # too large to journal
            return [] if change.op == "append" else [("delete", (log_key(ek), None))]
        member = change.entry_id + change.value
        return [("save" if change.op == "append" else "delete", (log_key(ek), member))]
    if change.field is None:
        return [("delete", (index_key(ek), None))]
    fk = field_key(ek, encode(change.field))
    if change.op == "put" and change.value is not None:
        return [("save", (fk, change.value))]
    return [("delete", (fk, None))]


class _JournalBatch(NamedTuple):
    """The result of a journal fetch, which is one of the journal hooks."""

    changes: list[Change]
    # The position after these changes, for each partition that was asked for.
    positions: dict[KeyPartT, bytes]
    # The partitions that the limit cut short, which may have more changes.
    cut: set[KeyPartT]
    # Anything the backend's wait needs, to tell whether a change has arrived
    # since this fetch.
    token: Any = None


def _read_journal(
    source: Any,
    cursors: Mapping[KeyPartT, bytes],
    timeout: float | None,
    limit: int | None,
) -> tuple[list[Change], dict[KeyPartT, bytes]]:
    """Read a journal, through the hooks ``_journal_fetch`` and ``_journal_wait``.

    This applies the cursor rules in section 4.4 of guides/database-design.md,
    so that backends need no bookkeeping. A cursor is the time it was issued,
    then a position that only the backend understands.
    """
    positions: dict[KeyPartT, bytes] = {}
    issued: dict[KeyPartT, float] = {}
    now = time.time()
    for partition, cursor in cursors.items():
        issued[partition], positions[partition] = split_cursor(cursor)
    expired = [
        p
        for p, issued_at in issued.items()
        if now - issued_at > source.journal_retention / 2
    ]
    if expired:
        raise JournalCursorExpired(expired)
    if not positions:
        return [], {}
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        batch = source._journal_fetch(positions, limit)
        positions = batch.positions
        remaining = None if deadline is None else deadline - time.monotonic()
        if batch.changes or (remaining is not None and remaining <= 0):
            # A cursor is issued afresh when its partition was read to the end.
            # One that the limit cut short keeps its old time, so that a reader
            # who falls behind is told, and does not miss changes silently.
            fresh = time.time()
            return batch.changes, {
                p: make_cursor(pos, issued[p] if p in batch.cut else fresh)
                for p, pos in positions.items()
            }
        source._journal_wait(batch, remaining)


def _unpack_position(fmt: str, position: bytes) -> tuple[Any, ...]:
    try:
        return struct.unpack(fmt, position)
    except struct.error:
        raise InvalidArgument(f"invalid journal position {position!r}") from None


class _JournalBuffer:
    """Changes for the partitions that someone is following, held in memory.

    This serves the journal for the in-memory database, for databases that
    emulate a journal with a change listener, and for the local database server.
    Positions name this buffer, so a cursor from another buffer is expired.
    """

    def __init__(self, retention: float, *, max_entries: int = 100_000) -> None:
        self.journal_retention = retention
        self.max_entries = max_entries
        self.token = os.urandom(8)
        self.cond = threading.Condition()
        self.seq = 0
        self.queues: dict[KeyPartT, list[tuple[int, float, Change]]] = {}
        self.dropped: dict[KeyPartT, int] = {}
        self.last_read: dict[KeyPartT, float] = {}
        self._next_expiry = 0.0

    def _position(self, seq: int) -> bytes:
        return self.token + seq.to_bytes(8, "big")

    def head(self, partition: KeyPartT) -> bytes:
        return make_cursor(self._journal_position(partition))

    def read(
        self,
        cursors: Mapping[KeyPartT, bytes],
        timeout: float | None,
        limit: int | None,
    ) -> tuple[list[Change], dict[KeyPartT, bytes]]:
        return _read_journal(self, cursors, timeout, limit)

    def add(self, changes: Iterable[Change]) -> None:
        now = time.monotonic()
        with self.cond:
            added = False
            for change in changes:
                queue = self.queues.get(change.key[0])
                if queue is not None:
                    self.seq += 1
                    queue.append((self.seq, now, change))
                    added = True
                    if len(queue) > self.max_entries:
                        self._drop(change.key[0], len(queue) - self.max_entries)
            if now >= self._next_expiry:
                self._expire(now)
            if added:
                self.cond.notify_all()

    def _drop(self, partition: KeyPartT, count: int) -> None:
        queue = self.queues[partition]
        self.dropped[partition] = queue[count - 1][0]
        del queue[:count]

    def _expire(self, now: float) -> None:
        retention = self.journal_retention
        self._next_expiry = now + min(1.0, retention / 10)
        for partition, queue in list(self.queues.items()):
            if now - self.last_read.get(partition, now) > retention:
                del self.queues[partition]
                self.dropped.pop(partition, None)
                self.last_read.pop(partition, None)
                continue
            stale = bisect_left(queue, now - retention, key=lambda e: e[1])
            if stale:
                self._drop(partition, stale)

    def expire_partitions(self, partitions: Iterable[KeyPartT]) -> None:
        """Expire every cursor for these partitions, because changes were lost."""
        with self.cond:
            self.seq += 1
            for partition in partitions:
                if partition in self.queues:
                    self.queues[partition].clear()
                    self.dropped[partition] = self.seq
            self.cond.notify_all()

    # The journal hooks. See _read_journal.

    def _journal_position(self, partition: KeyPartT) -> bytes:
        with self.cond:
            self.queues.setdefault(partition, [])
            self.last_read[partition] = time.monotonic()
            return self._position(self.seq)

    def _journal_fetch(
        self, positions: Mapping[KeyPartT, bytes], limit: int | None
    ) -> _JournalBatch:
        with self.cond:
            expired = []
            changes: list[Change] = []
            new_positions = {}
            cut = set()
            now = time.monotonic()
            for partition, position in positions.items():
                seq = int.from_bytes(position[8:16], "big")
                queue = self.queues.get(partition)
                if (
                    queue is None
                    or position[:8] != self.token
                    or seq < self.dropped.get(partition, 0)
                ):
                    expired.append(partition)
                    continue
                self.last_read[partition] = now
                if not queue or queue[-1][0] <= seq:
                    new_positions[partition] = self._position(self.seq)
                    continue
                start = bisect_right(queue, seq, key=lambda e: e[0])
                stop = len(queue)
                if limit is not None:
                    stop = min(stop, start + max(0, limit - len(changes)))
                changes.extend(change for _, _, change in queue[start:stop])
                if stop < len(queue):
                    cut.add(partition)
                    last = queue[stop - 1][0] if stop > start else seq
                    new_positions[partition] = self._position(last)
                else:
                    new_positions[partition] = self._position(self.seq)
            if expired:
                raise JournalCursorExpired(expired)
            return _JournalBatch(changes, new_positions, cut, self.seq)

    def _journal_wait(self, batch: _JournalBatch, timeout: float | None) -> None:
        with self.cond:
            if self.seq == batch.token:  # nothing has arrived since the fetch
                self.cond.wait(timeout)


class _ListenerThread(threading.Thread):
    """Gives every change in a store's journal to a database's listeners.

    Subclasses implement ``fetch``, which returns the changes after the ones it
    returned last, and ``wait``, which returns when there may be more.
    """

    def __init__(self, db: "ExampleDatabase") -> None:
        super().__init__(daemon=True, name="hypothesis-db-listener")
        self.db = db
        self.stopping = threading.Event()

    def fetch(self) -> list[Change]:
        raise NotImplementedError

    def wait(self) -> None:
        raise NotImplementedError

    def release(self) -> None:
        """Release what the thread holds. Called when it stops."""

    def run(self) -> None:
        try:
            while not self.stopping.is_set():
                changes = self.fetch()
                for change in changes:
                    for event in _events_from_change(change):
                        self.db._broadcast_change(event)
                if not changes:
                    self.wait()
        finally:
            self.release()

    def stop(self) -> None:
        self.stopping.set()
        self.join()


class _EmulationState:
    """Per-instance state for databases that emulate the structured API."""

    init_lock = threading.Lock()

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.last_id: bytes | None = None
        self.appends_since_trim: dict[bytes, int] = {}

    @classmethod
    def of(cls, db: "ExampleDatabase") -> "_EmulationState":
        try:
            return db.__dict__["_emulation_state"]
        except KeyError:
            with cls.init_lock:
                return db.__dict__.setdefault("_emulation_state", cls())

    def new_entry_id(self) -> bytes:
        # Ids increase within this process. The random sequence number makes them
        # unique across processes, but ids from different processes in the same
        # millisecond are not ordered.
        with self.lock:
            now_ms = int(time.time() * 1000)
            last_ms = split_entry_id(self.last_id)[0] if self.last_id else -1
            if now_ms > last_ms:
                self.last_id = make_entry_id(now_ms, random.getrandbits(48))
            else:
                self.last_id = next_entry_id(self.last_id, now_ms)
            return self.last_id


def _pick(members: Iterable[bytes]) -> bytes | None:
    # Concurrent puts can leave more than one value in a field key. Any
    # deterministic choice will do, and the next put removes the others.
    return max(members, default=None)


def _emulated_log_entries(
    db: "ExampleDatabase", ek: bytes
) -> list[tuple[bytes, bytes]]:
    members = {m for m in db.fetch(log_key(ek)) if len(m) >= ENTRY_ID_SIZE}
    return sorted((m[:ENTRY_ID_SIZE], m[ENTRY_ID_SIZE:]) for m in members)


def _emulated_read(db: "ExampleDatabase", op: ReadOpT) -> Any:
    if isinstance(op, LogRange):
        return _select_entries(_emulated_log_entries(db, encode(op.key)), op)
    if is_legacy(op.key):
        members = set(db.fetch(cast(bytes, op.key[0])))
        if isinstance(op, MapGet):
            return b"" if op.field[0] in members else None
        return {(m,): b"" for m in members if _matches_prefix((m,), op.prefix)}
    ek = encode(op.key)
    if isinstance(op, MapGet):
        return _pick(db.fetch(field_key(ek, encode(op.field))))
    prefix = encode(op.prefix)
    result = {}
    for ef in set(db.fetch(index_key(ek))):
        if (
            ef.startswith(prefix)
            and (value := _pick(db.fetch(field_key(ek, ef)))) is not None
        ):
            try:
                result[decode(ef)] = value
            except ValueError:
                continue
    return result


def _emulated_current(db: "ExampleDatabase", op: MapPut | MapDelete) -> bytes | None:
    return _emulated_read(db, MapGet(op.key, op.field))


def _emulated_trim(db: "ExampleDatabase", ek: bytes, op: LogTrim) -> int:
    entries = _emulated_log_entries(db, ek)
    keep = [
        e
        for e in entries
        if e[0] not in op.ids and (op.before is None or e[0] >= op.before)
    ]
    if op.maxlen is not None:
        keep = keep[max(0, len(keep) - op.maxlen) :]
    doomed = set(entries) - set(keep)
    for entry_id, value in doomed:
        db.delete(log_key(ek), entry_id + value)
    return len(doomed)


def _emulated_write(db: "ExampleDatabase", op: WriteOpT, *, check: bool) -> Any:
    if isinstance(op, (MapPut, MapDelete)):
        if check and _is_conditional(op) and _emulated_current(db, op) != op.expect:
            return False
        if is_legacy(op.key):
            raw, member = cast(bytes, op.key[0]), cast(bytes, op.field[0])
            if isinstance(op, MapPut):
                db.save(raw, member)
                return True
            present = member in set(db.fetch(raw))
            if present:
                db.delete(raw, member)
            return present
        ek, ef = encode(op.key), encode(op.field)
        fk = field_key(ek, ef)
        members = set(db.fetch(fk))
        if isinstance(op, MapPut):
            # Write the index entry first, so that a crash leaves an index entry
            # with no value, which reads as absent.
            db.save(index_key(ek), ef)
            if op.value not in members:
                db.save(fk, op.value)
            members.discard(op.value)
        for member in members:
            db.delete(fk, member)
        if isinstance(op, MapPut):
            return True
        db.delete(index_key(ek), ef)
        return bool(members)
    if isinstance(op, MapClear):
        if is_legacy(op.key):
            raw = cast(bytes, op.key[0])
            for member in list(db.fetch(raw)):
                db.delete(raw, member)
            return None
        ek = encode(op.key)
        for ef in list(db.fetch(index_key(ek))):
            for member in list(db.fetch(field_key(ek, ef))):
                db.delete(field_key(ek, ef), member)
            db.delete(index_key(ek), ef)
        return None
    ek = encode(op.key)
    if isinstance(op, LogTrim):
        return _emulated_trim(db, ek, op)
    state = _EmulationState.of(db)
    entry_id = state.new_entry_id()
    db.save(log_key(ek), entry_id + op.value)
    if op.maxlen is not None:
        # Trimming reads the whole log, so do it after every maxlen/4 appends
        # from this process, rather than after every append.
        lk = log_key(ek)
        count = state.appends_since_trim.get(lk, 0) + 1
        if count >= max(1, op.maxlen // 4):
            _emulated_trim(db, ek, LogTrim(op.key, maxlen=op.maxlen))
            count = 0
        state.appends_since_trim[lk] = count
    return entry_id


def _usable_dir(path: StrPathT) -> bool:
    """
    Returns True if the desired path can be used as database path because
    either the directory exists and can be used, or its root directory can
    be used and we can make the directory as needed.
    """
    path = Path(path)
    try:
        while not path.exists():
            # Loop terminates because the root dir ('/' on unix) always exists.
            path = path.parent
        return path.is_dir() and os.access(path, os.R_OK | os.W_OK | os.X_OK)
    except PermissionError:  # pragma: no cover
        # path.exists() returns False on 3.14+ instead of raising. See
        # https://docs.python.org/3.14/library/pathlib.html#querying-file-type-and-status
        return False


def _db_for_path(
    path: StrPathT | UniqueIdentifier | Literal[":memory:"] | None = None,
) -> "ExampleDatabase":
    if path is not_set:
        if os.getenv("HYPOTHESIS_DATABASE_FILE") is not None:  # pragma: no cover
            raise HypothesisException(
                "The $HYPOTHESIS_DATABASE_FILE environment variable no longer has any "
                "effect.  Configure your database location via a settings profile instead.\n"
                "https://hypothesis.readthedocs.io/en/latest/settings.html#settings-profiles"
            )

        storage_dir = storage_directory("examples", intent_to_write=False)
        if not _usable_dir(storage_dir.path):  # pragma: no cover
            warnings.warn(
                "The database setting is not configured, and the default "
                "location is unusable - falling back to an in-memory "
                f"database for this session.  path={storage_dir.path!r}",
                HypothesisWarning,
                stacklevel=3,
            )
            return InMemoryExampleDatabase()
        return _StorageDirectoryDatabase(storage_dir)
    if path in (None, ":memory:"):
        return InMemoryExampleDatabase()
    path = cast(StrPathT, path)
    return DirectoryBasedExampleDatabase(path)


class _EDMeta(abc.ABCMeta):
    def __call__(self, *args: Any, **kwargs: Any) -> "ExampleDatabase":
        if self is ExampleDatabase:
            note_deprecation(
                "Creating a database using the abstract ExampleDatabase() class "
                "is deprecated. Prefer using a concrete subclass, like "
                "InMemoryExampleDatabase() or DirectoryBasedExampleDatabase(path). "
                'In particular, the special string ExampleDatabase(":memory:") '
                "should be replaced by InMemoryExampleDatabase().",
                since="2025-04-07",
                has_codemod=False,
            )
            return _db_for_path(*args, **kwargs)
        return super().__call__(*args, **kwargs)


# This __call__ method is picked up by Sphinx as the signature of all ExampleDatabase
# subclasses, which is accurate, reasonable, and unhelpful.  Fortunately Sphinx
# maintains a list of metaclass-call-methods to ignore, and while they would prefer
# not to maintain it upstream (https://github.com/sphinx-doc/sphinx/pull/8262) we
# can insert ourselves here.
#
# This code only runs if Sphinx has already been imported; and it would live in our
# docs/conf.py except that we would also like it to work for anyone documenting
# downstream ExampleDatabase subclasses too.
#
# We avoid type-checking this block due to this combination facts:
# * our check-types-api CI job runs under 3.14
# * tools.txt therefore pins to a newer version of sphinx which uses 3.12+ `type`
#   syntax
# * in test_mypy.py, mypy sees this block, sees sphinx is installed, tries parsing
#   sphinx code, and errors
#
# Putting `and not TYPE_CHECKING` here is just a convenience for our testing setup
# (because we don't split mypy tests by running CI version, eg), not for runtime
#  behavior.
if "sphinx" in sys.modules and not TYPE_CHECKING:  # pragma: no cover
    try:
        import sphinx.ext.autodoc

        signature = "hypothesis.database._EDMeta.__call__"

        # _METACLASS_CALL_BLACKLIST moved in newer sphinx versions
        try:
            import sphinx.ext.autodoc._dynamic._signatures as _module
        except ImportError:
            _module = sphinx.ext.autodoc

        # _METACLASS_CALL_BLACKLIST is a frozenset in later sphinx versions
        if isinstance(_module._METACLASS_CALL_BLACKLIST, frozenset):
            _module._METACLASS_CALL_BLACKLIST = _module._METACLASS_CALL_BLACKLIST | {
                signature
            }
        else:
            _module._METACLASS_CALL_BLACKLIST.append(signature)
    except Exception:
        pass


class ExampleDatabase(metaclass=_EDMeta):
    """
    A Hypothesis database, for use in |settings.database|.

    Hypothesis automatically saves failures to the database set in
    |settings.database|. The next time the test is run, Hypothesis will replay
    any failures from the database in |settings.database| for that test (in
    |Phase.reuse|).

    The database is best thought of as a cache that you never need to invalidate.
    Entries may be transparently dropped when upgrading your Hypothesis version
    or changing your test. Do not rely on the database for correctness; to ensure
    Hypothesis always tries an input, use |@example|.

    A Hypothesis database is a simple mapping of bytes to sets of bytes. Hypothesis
    provides several concrete database subclasses. To write your own database class,
    see :doc:`/how-to/custom-database`.

    Change listening
    ----------------

    An optional extension to |ExampleDatabase| is change listening. On databases
    which support change listening, calling |ExampleDatabase.add_listener| adds
    a function as a change listener, which will be called whenever a value is
    added, deleted, or moved inside the database. See |ExampleDatabase.add_listener|
    for details.

    All databases in Hypothesis support change listening. Custom database classes
    are not required to support change listening, though they will not be compatible
    with features that require change listening until they do so.

    .. note::

        While no Hypothesis features currently require change listening, change
        listening is required by `HypoFuzz <https://hypofuzz.com/>`_.

    Database methods
    ----------------

    Required methods:

    * |ExampleDatabase.save|
    * |ExampleDatabase.fetch|
    * |ExampleDatabase.delete|

    Optional methods:

    * |ExampleDatabase.move|

    Change listening methods:

    * |ExampleDatabase.add_listener|
    * |ExampleDatabase.remove_listener|
    * |ExampleDatabase.clear_listeners|
    * |ExampleDatabase._start_listening|
    * |ExampleDatabase._stop_listening|
    * |ExampleDatabase._broadcast_change|

    Structured data
    ---------------

    Every database also stores maps, logs, and a change journal, for tools such as
    HypoFuzz. Keys are tuples, and the first component of a key is its partition.
    A key with a single bytes component holds a set, as the methods above do.

    A database that implements only ``save``, ``fetch``, and ``delete`` supports
    all of this through emulation, which is slower and not atomic. A backend can
    instead implement ``read_many``, ``write_many``, ``journal_head``,
    ``journal_read``, and ``capabilities``. See ``guides/database-design.md``.
    """

    #: How long, in seconds, journal entries are kept after they are written.
    journal_retention: float = 300.0

    def __init__(self) -> None:
        self._listeners: list[ListenerT] = []

    @abc.abstractmethod
    def save(self, key: bytes, value: bytes) -> None:
        """Save ``value`` under ``key``.

        If ``value`` is already present in ``key``, silently do nothing.
        """
        raise NotImplementedError(f"{type(self).__name__}.save")

    @abc.abstractmethod
    def fetch(self, key: bytes) -> Iterable[bytes]:
        """Return an iterable over all values matching this key."""
        raise NotImplementedError(f"{type(self).__name__}.fetch")

    @abc.abstractmethod
    def delete(self, key: bytes, value: bytes) -> None:
        """Remove ``value`` from ``key``.

        If ``value`` is not present in ``key``, silently do nothing.
        """
        raise NotImplementedError(f"{type(self).__name__}.delete")

    def move(self, src: bytes, dest: bytes, value: bytes) -> None:
        """
        Move ``value`` from key ``src`` to key ``dest``.

        Equivalent to ``delete(src, value)`` followed by ``save(src, value)``,
        but may have a more efficient implementation.

        Note that ``value`` will be inserted at ``dest`` regardless of whether
        it is currently present at ``src``.
        """
        if src == dest:
            self.save(src, value)
            return
        self.delete(src, value)
        self.save(dest, value)

    def add_listener(self, f: ListenerT, /) -> None:
        """
        Add a change listener. ``f`` will be called whenever a value is saved,
        deleted, or moved in the database.

        ``f`` can be called with two different event values:

        * ``("save", (key, value))``
        * ``("delete", (key, value))``

        where ``key`` and ``value`` are both ``bytes``.

        There is no ``move`` event. Instead, a move is broadcasted as a
        ``delete`` event followed by a ``save`` event.

        For the ``delete`` event, ``value`` may be ``None``. This might occur if
        the database knows that a deletion has occurred in ``key``, but does not
        know what value was deleted.
        """
        was_listening = self._wants_events()
        self._listeners.append(f)
        if not was_listening:
            self._start_listening()

    def remove_listener(self, f: ListenerT, /) -> None:
        """
        Removes ``f`` from the list of change listeners.

        If ``f`` is not in the list of change listeners, silently do nothing.
        """
        if f not in self._listeners:
            return
        self._listeners.remove(f)
        if not self._wants_events():
            self._stop_listening()

    def clear_listeners(self) -> None:
        """Remove all change listeners."""
        had_listeners = bool(self._listeners)
        self._listeners.clear()
        if had_listeners and not self._wants_events():
            self._stop_listening()

    def _wants_events(self) -> bool:
        return bool(self._listeners) or "_journal_buffer" in self.__dict__

    def _broadcast_change(self, event: ListenerEventT) -> None:
        """
        Called when a value has been either added to or deleted from a key in
        the underlying database store. The possible values for ``event`` are:

        * ``("save", (key, value))``
        * ``("delete", (key, value))``

        ``value`` may be ``None`` for the ``delete`` event, indicating we know
        that some value was deleted under this key, but not its exact value.

        Note that you should not assume your instance is the only reference to
        the underlying database store. For example, if two instances of
        |DirectoryBasedExampleDatabase| reference the same directory,
        _broadcast_change should be called whenever a file is added or removed
        from the directory, even if that database was not responsible for
        changing the file.
        """
        for listener in self._listeners:
            listener(event)
        journal = self.__dict__.get("_journal_buffer")
        if journal is not None and (change := _change_from_event(event)) is not None:
            journal.add([change])

    def _start_listening(self) -> None:
        """
        Called when the database adds a change listener, and did not previously
        have any change listeners. Intended to allow databases to wait to start
        expensive listening operations until necessary.

        ``_start_listening`` and ``_stop_listening`` are guaranteed to alternate,
        so you do not need to handle the case of multiple consecutive
        ``_start_listening`` calls without an intermediate ``_stop_listening``
        call.
        """
        warnings.warn(
            f"{self.__class__} does not support listening for changes",
            HypothesisWarning,
            stacklevel=4,
        )

    def _stop_listening(self) -> None:
        """
        Called whenever no change listeners remain on the database.

        ``_stop_listening`` and ``_start_listening`` are guaranteed to alternate,
        so you do not need to handle the case of multiple consecutive
        ``_stop_listening`` calls without an intermediate ``_start_listening``
        call.
        """
        warnings.warn(
            f"{self.__class__} does not support stopping listening for changes",
            HypothesisWarning,
            stacklevel=4,
        )

    # Structured data. The single-operation methods are shorthand for read_many
    # and write_many, which backends override. See guides/database-design.md.

    @property
    def capabilities(self) -> frozenset[str]:
        """What this database guarantees, as a subset of ``native``, ``atomic``,
        ``journal``, ``blocking``, ``shared``, ``ttl``, and ``server_time``."""
        return frozenset()

    def map_get(self, key: KeyT, field: KeyT) -> bytes | None:
        """Return the value of ``field`` in the map at ``key``, or ``None``."""
        return self.read_many([MapGet(key, field)])[0]

    def map_items(self, key: KeyT, *, prefix: KeyT = ()) -> dict[KeyTupleT, bytes]:
        """Return the fields of the map at ``key`` that extend ``prefix``."""
        return self.read_many([MapItems(key, prefix=prefix)])[0]

    def map_put(
        self,
        key: KeyT,
        field: KeyT,
        value: bytes = b"",
        *,
        ttl: TTLT = None,
        expect: bytes | _Unset | None = unset,
    ) -> bool | None:
        """Set ``field`` to ``value`` in the map at ``key``.

        With ``expect=None`` the write applies only if the field is absent, and
        with ``expect=b"..."`` only if the field has that value. The entry may
        disappear at any time after ``ttl`` seconds.

        Returns ``True`` if the write applied, ``False`` if a condition failed,
        and ``None`` if the database queued the write to apply later.
        """
        return self.write_many([MapPut(key, field, value, ttl=ttl, expect=expect)])[0]

    def map_delete(
        self, key: KeyT, field: KeyT, *, expect: bytes | _Unset = unset
    ) -> bool | None:
        """Delete ``field`` from the map at ``key``.

        Returns ``True`` if the field existed and matched ``expect``, and ``None``
        if the database queued the write to apply later.
        """
        return self.write_many([MapDelete(key, field, expect=expect)])[0]

    def map_clear(self, key: KeyT) -> None:
        """Delete every field of the map at ``key``."""
        self.write_many([MapClear(key)])

    def log_append(
        self, key: KeyT, value: bytes, *, maxlen: int | None = None, ttl: TTLT = None
    ) -> bytes | None:
        """Append ``value`` to the log at ``key``, and return the new entry's id.

        The log keeps at least the newest ``maxlen`` entries. Entries older than
        ``ttl`` seconds may be removed. Returns ``None`` if the database queued
        the write to apply later.
        """
        return self.write_many([LogAppend(key, value, maxlen=maxlen, ttl=ttl)])[0]

    def log_range(
        self,
        key: KeyT,
        *,
        after: bytes | None = None,
        before: bytes | None = None,
        limit: int | None = None,
        reverse: bool = False,
    ) -> list[tuple[bytes, bytes]]:
        """Return ``(entry_id, value)`` pairs from the log at ``key``.

        Entry ids increase in the order the entries were appended. ``after``
        and ``before`` are exclusive bounds.
        """
        op = LogRange(key, after=after, before=before, limit=limit, reverse=reverse)
        return self.read_many([op])[0]

    def log_trim(
        self,
        key: KeyT,
        *,
        maxlen: int | None = None,
        before: bytes | None = None,
        ids: Iterable[bytes] = (),
    ) -> int | None:
        """Remove the entries whose ids are in ``ids``, and the entries before
        ``before``, and then all but the newest ``maxlen`` entries.

        Returns the number of entries removed, or ``None`` if the database
        queued the write to apply later.
        """
        return self.write_many([LogTrim(key, maxlen=maxlen, before=before, ids=ids)])[0]

    def read_many(self, ops: Sequence[ReadOpT]) -> list[Any]:
        """Run several reads, in one round trip where the backend allows."""
        return [_emulated_read(self, op) for op in ops]

    def write_many(self, ops: Sequence[WriteOpT], *, atomic: bool = False) -> list[Any]:
        """Apply several writes in order, and return their results.

        With ``atomic=True``, every operation must be in the same partition, and
        either all of them apply or none do. This emulation is not atomic.
        """
        ops = list(ops)
        if atomic:
            _check_atomic(ops)
            if not all(
                _emulated_current(self, op) == op.expect
                for op in ops
                if _is_conditional(op)
            ):
                return _not_applied(ops)
        return [_emulated_write(self, op, check=not atomic) for op in ops]

    def journal_head(self, partition: KeyPartT) -> bytes:
        """Return a cursor for the current end of ``partition``'s journal."""
        return make_cursor(self._journal_position(partition))

    def journal_read(
        self,
        cursors: Mapping[KeyPartT, bytes],
        *,
        timeout: float | None = 0,
        limit: int | None = None,
    ) -> tuple[list[Change], dict[KeyPartT, bytes]]:
        """Return changes after each cursor, and new cursors for every partition.

        Changes arrive at least once, in order within each partition. With a
        positive ``timeout``, wait up to that many seconds for a change, or
        forever if ``timeout`` is ``None``. Raises :class:`JournalCursorExpired`
        if changes may have been lost.
        """
        return _read_journal(self, cursors, timeout, limit)

    # The journal hooks, which backends override. By default, the journal is
    # built from the change listener, so it sees only the changes that the
    # listener reports, and only from when the first cursor was taken.

    def _journal_position(self, partition: KeyPartT) -> bytes:
        """The position at the current end of ``partition``'s journal."""
        return self._listener_journal()._journal_position(partition)

    def _journal_fetch(
        self, positions: Mapping[KeyPartT, bytes], limit: int | None
    ) -> _JournalBatch:
        """The changes after each position, at most ``limit`` in all."""
        return self._listener_journal()._journal_fetch(positions, limit)

    def _journal_wait(self, batch: _JournalBatch, timeout: float | None) -> None:
        """Return when a change may have arrived since ``batch`` was fetched, or
        after ``timeout`` seconds. Returning early does no harm."""
        self._listener_journal()._journal_wait(batch, timeout)

    def _listener_journal(self) -> _JournalBuffer:
        # A journal built from the change listener. It sees only the changes that
        # the listener reports, and only from when the first cursor was taken.
        try:
            return self.__dict__["_journal_buffer"]
        except KeyError:
            with _EmulationState.init_lock:
                if "_journal_buffer" in self.__dict__:
                    return self.__dict__["_journal_buffer"]
                was_listening = self._wants_events()
                journal = self.__dict__["_journal_buffer"] = _JournalBuffer(
                    self.journal_retention
                )
            if not was_listening:
                self._start_listening()
            return journal

    def current_time(self) -> float:
        """The store's clock, in seconds since the epoch."""
        return time.time()

    def flush(self, timeout: float | None = None) -> None:
        """Wait until every queued write has been applied."""

    def close(self) -> None:
        """Release the connections and threads that this database holds.

        This removes every listener. Using the database again opens what it needs.
        """
        was_listening = self._wants_events()
        self._listeners.clear()
        self.__dict__.pop("_journal_buffer", None)
        if was_listening:
            self._stop_listening()

    def __enter__(self: _DatabaseT) -> _DatabaseT:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class _NativeDatabase(ExampleDatabase):
    """A backend that implements the structured API directly.

    The old methods become views onto the structured data. Keys passed to
    ``save``, ``fetch`` and ``delete`` are sets, unless they are one of the
    reserved keys that the emulation uses, which map onto fields and logs.
    """

    @abc.abstractmethod
    def read_many(self, ops: Sequence[ReadOpT]) -> list[Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def write_many(self, ops: Sequence[WriteOpT], *, atomic: bool = False) -> list[Any]:
        raise NotImplementedError

    #: A thread that gives the store's changes to listeners, if the backend has one.
    _listener_thread_class: ClassVar[type[_ListenerThread] | None] = None

    def _start_listening(self) -> None:
        if self._listener_thread_class is None:
            super()._start_listening()
            return
        thread = self.__dict__["_listener_thread"] = self._listener_thread_class(self)
        thread.start()

    def _stop_listening(self) -> None:
        thread = self.__dict__.pop("_listener_thread", None)
        if thread is None:
            super()._stop_listening()
        else:
            thread.stop()

    def save(self, key: bytes, value: bytes) -> None:
        parsed = parse_legacy_key(key)
        if parsed.kind == "set":
            self.write_many([MapPut(parsed.key, (bytes(value),))])
        elif parsed.kind == "field":
            self.write_many([MapPut(parsed.key, parsed.field, value)])
        elif parsed.kind == "log" and len(value) >= ENTRY_ID_SIZE:
            # The store assigns its own ids.
            self.write_many([LogAppend(parsed.key, value[ENTRY_ID_SIZE:])])

    def fetch(self, key: bytes) -> Iterable[bytes]:
        # Read now, but return an iterator, as the old implementations did.
        parsed = parse_legacy_key(key)
        if parsed.kind == "set":
            return iter([cast(bytes, f[0]) for f in self.map_items(parsed.key)])
        if parsed.kind == "field":
            value = self.map_get(parsed.key, parsed.field)
            return iter([] if value is None else [value])
        if parsed.kind == "index":
            return iter([encode(f) for f in self.map_items(parsed.key)])
        return iter(
            [entry_id + value for entry_id, value in self.log_range(parsed.key)]
        )

    def delete(self, key: bytes, value: bytes) -> None:
        parsed = parse_legacy_key(key)
        if parsed.kind == "set":
            self.write_many([MapDelete(parsed.key, (bytes(value),))])
        elif parsed.kind == "field":
            self.write_many([MapDelete(parsed.key, parsed.field, expect=value)])
        elif parsed.kind == "log" and len(value) >= ENTRY_ID_SIZE:
            self.write_many([LogTrim(parsed.key, ids=[value[:ENTRY_ID_SIZE]])])

    def move(self, src: bytes, dest: bytes, value: bytes) -> None:
        if src == dest:
            self.save(src, value)
            return
        source, target = parse_legacy_key(src), parse_legacy_key(dest)
        if source.kind == target.kind == "set":
            self.write_many(
                [
                    MapDelete(source.key, (bytes(value),)),
                    MapPut(target.key, (bytes(value),)),
                ]
            )
        else:
            super().move(src, dest, value)


class _MemoryLog:
    __slots__ = ("entries", "last_id", "ttl")

    def __init__(self) -> None:
        self.entries: list[tuple[bytes, bytes]] = []
        self.last_id: bytes | None = None
        self.ttl: float | None = None

    def live(self, now: float) -> list[tuple[bytes, bytes]]:
        if self.ttl is not None and self.entries:
            cutoff = make_entry_id(max(0, int((now - self.ttl) * 1000)), 0)
            if stale := bisect_left(self.entries, cutoff, key=lambda e: e[0]):
                del self.entries[:stale]
        return self.entries


class InMemoryExampleDatabase(_NativeDatabase):
    """A non-persistent example database, implemented in terms of an in-memory
    dictionary.

    This can be useful if you call a test function several times in a single
    session, or for testing other database implementations, but because it
    does not persist between runs we do not recommend it for general use.
    """

    def __init__(self) -> None:
        super().__init__()
        self.data: dict[bytes, set[bytes]] = {}
        self._maps: dict[KeyTupleT, dict[KeyTupleT, tuple[bytes, float | None]]] = {}
        self._logs: dict[KeyTupleT, _MemoryLog] = {}
        self._lock = threading.RLock()
        self._journal = _JournalBuffer(self.journal_retention)

    def __repr__(self) -> str:
        return f"InMemoryExampleDatabase({self.data!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, InMemoryExampleDatabase) and self.data is other.data

    @property
    def capabilities(self) -> frozenset[str]:
        return frozenset({"native", "atomic", "journal", "blocking", "ttl"})

    def _live_value(self, key: KeyTupleT, field: KeyTupleT, now: float) -> bytes | None:
        if is_legacy(key):
            return b"" if field[0] in self.data.get(cast(bytes, key[0]), ()) else None
        entry = self._maps.get(key, {}).get(field)
        if entry is None or (entry[1] is not None and entry[1] <= now):
            return None
        return entry[0]

    def read_many(self, ops: Sequence[ReadOpT]) -> list[Any]:
        now = time.time()
        with self._lock:
            return [self._read(op, now) for op in ops]

    def _read(self, op: ReadOpT, now: float) -> Any:
        if isinstance(op, MapGet):
            return self._live_value(op.key, op.field, now)
        if isinstance(op, LogRange):
            log = self._logs.get(op.key)
            return _select_entries(log.live(now) if log else [], op)
        if is_legacy(op.key):
            members = self.data.get(cast(bytes, op.key[0]), ())
            return {(m,): b"" for m in members if _matches_prefix((m,), op.prefix)}
        return {
            field: value
            for field, (value, expires) in self._maps.get(op.key, {}).items()
            if (expires is None or expires > now) and _matches_prefix(field, op.prefix)
        }

    def write_many(self, ops: Sequence[WriteOpT], *, atomic: bool = False) -> list[Any]:
        ops = list(ops)
        if atomic:
            _check_atomic(ops)
        now = time.time()
        changes: list[Change] = []
        with self._lock:
            if atomic and not all(
                self._live_value(op.key, op.field, now) == op.expect
                for op in ops
                if _is_conditional(op)
            ):
                return _not_applied(ops)
            results = [self._write(op, now, changes, check=not atomic) for op in ops]
            if changes:
                self._journal.add(changes)
        # Call listeners outside the lock, as the old implementation did.
        if changes and self._listeners:
            for change in changes:
                for event in _events_from_change(change):
                    self._broadcast_change(event)
        return results

    def _write(
        self, op: WriteOpT, now: float, changes: list[Change], *, check: bool
    ) -> Any:
        if isinstance(op, (MapPut, MapDelete)):
            current = self._live_value(op.key, op.field, now)
            if check and _is_conditional(op) and current != op.expect:
                return False
            if isinstance(op, MapPut):
                if is_legacy(op.key):
                    self.data.setdefault(cast(bytes, op.key[0]), set()).add(
                        cast(bytes, op.field[0])
                    )
                else:
                    expires = None if op.ttl is None else now + op.ttl
                    self._maps.setdefault(op.key, {})[op.field] = (op.value, expires)
                if current != op.value:
                    changes.append(Change("put", op.key, op.field, value=op.value))
                return True
            if current is None:
                return False
            if is_legacy(op.key):
                self.data[cast(bytes, op.key[0])].discard(cast(bytes, op.field[0]))
            else:
                del self._maps[op.key][op.field]
            changes.append(Change("delete", op.key, op.field))
            return True
        if isinstance(op, MapClear):
            if is_legacy(op.key):
                members = self.data.get(cast(bytes, op.key[0]))
                cleared = bool(members)
                if members:
                    members.clear()
            else:
                cleared = bool(self._read(MapItems(op.key), now))
                self._maps.pop(op.key, None)
            if cleared:
                changes.append(Change("clear", op.key))
            return None
        log = self._logs.setdefault(op.key, _MemoryLog())
        if isinstance(op, LogTrim):
            count = len(log.entries)
            if op.ids:
                ids = set(op.ids)
                kept = []
                for entry in log.entries:
                    if entry[0] in ids:
                        changes.append(
                            Change("delete", op.key, entry_id=entry[0], value=entry[1])
                        )
                    else:
                        kept.append(entry)
                log.entries = kept
            if op.before is not None:
                del log.entries[
                    : bisect_left(log.entries, op.before, key=lambda e: e[0])
                ]
            if op.maxlen is not None:
                del log.entries[: max(0, len(log.entries) - op.maxlen)]
            return count - len(log.entries)
        entry_id = next_entry_id(log.last_id, int(now * 1000))
        log.entries.append((entry_id, op.value))
        log.last_id = entry_id
        log.ttl = op.ttl
        if op.maxlen is not None and len(log.entries) > op.maxlen:
            del log.entries[: len(log.entries) - op.maxlen]
        changes.append(Change("append", op.key, entry_id=entry_id, value=op.value))
        return entry_id

    def _journal_position(self, partition: KeyPartT) -> bytes:
        return self._journal._journal_position(partition)

    def _journal_fetch(
        self, positions: Mapping[KeyPartT, bytes], limit: int | None
    ) -> _JournalBatch:
        return self._journal._journal_fetch(positions, limit)

    def _journal_wait(self, batch: _JournalBatch, timeout: float | None) -> None:
        self._journal._journal_wait(batch, timeout)

    def _start_listening(self) -> None:
        # Listeners are called directly by write_many, since every write to
        # this database goes through this object.
        pass

    def _stop_listening(self) -> None:
        pass


def _hash(key: bytes) -> str:
    return sha384(key).hexdigest()[:16]


class DirectoryBasedExampleDatabase(ExampleDatabase):
    """Use a directory to store Hypothesis examples as files.

    Each test corresponds to a directory, and each example to a file within that
    directory.  While the contents are fairly opaque, a
    |DirectoryBasedExampleDatabase| can be shared by checking the directory
    into version control, for example with the following ``.gitignore``::

        # Ignore files cached by Hypothesis...
        .hypothesis/*
        # except for the examples directory
        !.hypothesis/examples/

    Note however that this only makes sense if you also pin to an exact version of
    Hypothesis, and we would usually recommend implementing a shared database with
    a network datastore - see |ExampleDatabase|, and the |MultiplexedDatabase| helper.
    """

    # we keep a database entry of the full values of all the database keys.
    # currently only used for inverse mapping of hash -> key in change listening.
    _metakeys_name: ClassVar[bytes] = b".hypothesis-keys"
    _metakeys_hash: ClassVar[str] = _hash(_metakeys_name)

    def __init__(self, path: StrPathT) -> None:
        super().__init__()
        self.path = Path(path)
        self.keypaths: dict[bytes, Path] = {}
        self._observer: BaseObserver | None = None
        self._ensure_directory_exists_called = False

    def _ensure_directory_exists(self) -> None:
        # disk hits are expensive: early-return for performance
        if self._ensure_directory_exists_called:
            return

        self.path.mkdir(exist_ok=True, parents=True)
        self._ensure_directory_exists_called = True

    def __repr__(self) -> str:
        return f"DirectoryBasedExampleDatabase({self.path!r})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, DirectoryBasedExampleDatabase) and self.path == other.path
        )

    @property
    def capabilities(self) -> frozenset[str]:
        return frozenset({"journal"})

    def _key_path(self, key: bytes) -> Path:
        try:
            return self.keypaths[key]
        except KeyError:
            pass
        self.keypaths[key] = self.path / _hash(key)
        return self.keypaths[key]

    def _value_path(self, key: bytes, value: bytes) -> Path:
        return self._key_path(key) / _hash(value)

    def fetch(self, key: bytes) -> Iterable[bytes]:
        kp = self._key_path(key)
        if not kp.is_dir():
            return

        try:
            for path in os.listdir(kp):
                try:
                    yield (kp / path).read_bytes()
                except OSError:
                    pass
        except OSError:  # pragma: no cover
            # the `kp` directory might have been deleted in the meantime
            pass

    def save(self, key: bytes, value: bytes) -> None:
        key_path = self._key_path(key)
        if key_path.name != self._metakeys_hash:
            # add this key to our meta entry of all keys - taking care to avoid
            # infinite recursion.
            self.save(self._metakeys_name, key)

        # Note: we attempt to create the dir in question now. We
        # already checked for permissions, but there can still be other issues,
        # e.g. the disk is full, or permissions might have been changed.
        try:
            self._ensure_directory_exists()
            key_path.mkdir(exist_ok=True, parents=True)
            path = self._value_path(key, value)
            if not path.exists():
                # to mimic an atomic write, create and write in a temporary
                # directory, and only move to the final path after. This avoids
                # any intermediate state where the file is created (and empty)
                # but not yet written to.
                fd, tmpname = tempfile.mkstemp()
                tmppath = Path(tmpname)
                os.write(fd, value)
                os.close(fd)
                try:
                    tmppath.rename(path)
                except OSError as err:  # pragma: no cover
                    if err.errno == errno.EXDEV:
                        # Can't rename across filesystem boundaries, see e.g.
                        # https://github.com/HypothesisWorks/hypothesis/issues/4335
                        try:
                            path.write_bytes(tmppath.read_bytes())
                        except OSError:
                            pass
                    tmppath.unlink()
                assert not tmppath.exists()
        except OSError:  # pragma: no cover
            pass

    def move(self, src: bytes, dest: bytes, value: bytes) -> None:
        if src == dest:
            self.save(src, value)
            return

        src_path = self._value_path(src, value)
        dest_path = self._value_path(dest, value)
        # if the dest key path does not exist, os.renames will create it for us,
        # and we will never track its creation in the meta keys entry. Do so now.
        if not self._key_path(dest).exists():
            self.save(self._metakeys_name, dest)

        try:
            os.renames(src_path, dest_path)
        except OSError:
            self.delete(src, value)
            self.save(dest, value)

    def delete(self, key: bytes, value: bytes) -> None:
        try:
            self._value_path(key, value).unlink()
        except OSError:
            return

        # try deleting the key dir, which will only succeed if the dir is empty
        # (i.e. ``value`` was the last value in this key).
        try:
            self._key_path(key).rmdir()
        except OSError:
            pass
        else:
            # if the deletion succeeded, also delete this key entry from metakeys.
            # (if this key happens to be the metakey itself, this deletion will
            # fail; that's ok and faster than checking for this rare case.)
            self.delete(self._metakeys_name, key)

    def _start_listening(self) -> None:
        try:
            from watchdog.events import (
                DirCreatedEvent,
                DirDeletedEvent,
                DirMovedEvent,
                FileCreatedEvent,
                FileDeletedEvent,
                FileMovedEvent,
                FileSystemEventHandler,
            )
            from watchdog.observers import Observer
        except ImportError:
            warnings.warn(
                f"listening for changes in a {self.__class__.__name__} "
                "requires the watchdog library. To install, run "
                "`pip install hypothesis[watchdog]`",
                HypothesisWarning,
                stacklevel=4,
            )
            return

        hash_to_key = {_hash(key): key for key in self.fetch(self._metakeys_name)}
        _metakeys_hash = self._metakeys_hash
        _broadcast_change = self._broadcast_change

        class Handler(
            FileSystemEventHandler
        ):  # pragma: no cover # skipped in test_database.py for now
            def on_created(_self, event: FileCreatedEvent | DirCreatedEvent) -> None:
                # we only registered for the file creation event
                assert not isinstance(event, DirCreatedEvent)
                # watchdog events are only bytes if we passed a byte path to
                # .schedule
                assert isinstance(event.src_path, str)

                value_path = Path(event.src_path)
                # the parent dir represents the key, and its name is the key hash
                key_hash = value_path.parent.name

                if key_hash == _metakeys_hash:
                    try:
                        hash_to_key[value_path.name] = value_path.read_bytes()
                    except OSError:  # pragma: no cover
                        # this might occur if all the values in a key have been
                        # deleted and DirectoryBasedExampleDatabase removes its
                        # metakeys entry (which is `value_path` here`).
                        pass
                    return

                key = hash_to_key.get(key_hash)
                if key is None:  # pragma: no cover
                    # we didn't recognize this key. This shouldn't ever happen,
                    # but some race condition trickery might cause this.
                    return

                try:
                    value = value_path.read_bytes()
                except OSError:  # pragma: no cover
                    return

                _broadcast_change(("save", (key, value)))

            def on_deleted(self, event: FileDeletedEvent | DirDeletedEvent) -> None:
                assert not isinstance(event, DirDeletedEvent)
                assert isinstance(event.src_path, str)

                value_path = Path(event.src_path)
                key = hash_to_key.get(value_path.parent.name)
                if key is None:  # pragma: no cover
                    return

                _broadcast_change(("delete", (key, None)))

            def on_moved(self, event: FileMovedEvent | DirMovedEvent) -> None:
                assert not isinstance(event, DirMovedEvent)
                assert isinstance(event.src_path, str)
                assert isinstance(event.dest_path, str)

                src_path = Path(event.src_path)
                dest_path = Path(event.dest_path)
                k1 = hash_to_key.get(src_path.parent.name)
                k2 = hash_to_key.get(dest_path.parent.name)

                if k1 is None or k2 is None:  # pragma: no cover
                    return

                try:
                    value = dest_path.read_bytes()
                except OSError:  # pragma: no cover
                    return

                _broadcast_change(("delete", (k1, value)))
                _broadcast_change(("save", (k2, value)))

        # If we add a listener to a DirectoryBasedExampleDatabase whose database
        # directory doesn't yet exist, the watchdog observer will not fire any
        # events, even after the directory gets created.
        #
        # Ensure the directory exists before starting the observer.
        self._ensure_directory_exists()
        self._observer = Observer()
        self._observer.schedule(
            Handler(),
            # remove type: ignore when released
            # https://github.com/gorakhargosh/watchdog/pull/1096
            self.path,  # type: ignore
            recursive=True,
            event_filter=[FileCreatedEvent, FileDeletedEvent, FileMovedEvent],
        )
        self._observer.start()

    def _stop_listening(self) -> None:
        assert self._observer is not None
        self._observer.stop()
        self._observer.join()
        self._observer = None


class _StorageDirectoryDatabase(DirectoryBasedExampleDatabase):
    # A DirectoryBasedExampleDatabase which is located at the same directory as the storage
    # directory. This lets our database logic interact with our logic for writing .gitignore
    # files to the storage directory.
    #
    # The reason why we need this class is because the first interaction we have
    # with .hypothesis might be writing a file to .hypothesis/examples, and
    # DirectoryBasedExampleDatabase.save would otherwise create .hypothesis without
    # performing our .gitignore logic.

    def __init__(self, storage_dir: StorageDirectory) -> None:
        super().__init__(storage_dir.path)
        self._storage_dir = storage_dir

    def _ensure_directory_exists(self) -> None:
        if self._ensure_directory_exists_called:
            return

        self._storage_dir.create_if_missing()
        self._ensure_directory_exists_called = True


class ReadOnlyDatabase(ExampleDatabase):
    """A wrapper to make the given database read-only.

    The implementation passes through ``fetch``, and turns ``save``, ``delete``, and
    ``move`` into silent no-ops.

    Note that this disables Hypothesis' automatic discarding of stale examples.
    It is designed to allow local machines to access a shared database (e.g. from CI
    servers), without propagating changes back from a local or in-development branch.
    """

    def __init__(self, db: ExampleDatabase) -> None:
        super().__init__()
        assert isinstance(db, ExampleDatabase)
        self._wrapped = db

    def __repr__(self) -> str:
        return f"ReadOnlyDatabase({self._wrapped!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ReadOnlyDatabase) and self._wrapped == other._wrapped

    def fetch(self, key: bytes) -> Iterable[bytes]:
        yield from self._wrapped.fetch(key)

    def save(self, key: bytes, value: bytes) -> None:
        pass

    def delete(self, key: bytes, value: bytes) -> None:
        pass

    @property
    def capabilities(self) -> frozenset[str]:
        return self._wrapped.capabilities

    def read_many(self, ops: Sequence[ReadOpT]) -> list[Any]:
        return self._wrapped.read_many(ops)

    def write_many(self, ops: Sequence[WriteOpT], *, atomic: bool = False) -> list[Any]:
        return _not_applied(list(ops))

    def journal_head(self, partition: KeyPartT) -> bytes:
        return self._wrapped.journal_head(partition)

    def journal_read(
        self,
        cursors: Mapping[KeyPartT, bytes],
        *,
        timeout: float | None = 0,
        limit: int | None = None,
    ) -> tuple[list[Change], dict[KeyPartT, bytes]]:
        return self._wrapped.journal_read(cursors, timeout=timeout, limit=limit)

    def current_time(self) -> float:
        return self._wrapped.current_time()

    def close(self) -> None:
        super().close()
        self._wrapped.close()

    def _start_listening(self) -> None:
        # we're read only, so there are no changes to broadcast.
        pass

    def _stop_listening(self) -> None:
        pass


def _pack_cursors(cursors: list[bytes]) -> bytes:
    return make_cursor(b"".join(_pack_uleb128(len(c)) + c for c in cursors))


def _unpack_cursors(cursor: bytes, count: int) -> list[bytes]:
    _, data = split_cursor(cursor)
    parts = []
    try:
        while data:
            used, size = _unpack_uleb128(data)
            parts.append(data[used : used + size])
            data = data[used + size :]
    except ValueError:
        pass
    if len(parts) != count:
        raise InvalidArgument(f"invalid journal cursor {cursor!r}")
    return parts


class MultiplexedDatabase(ExampleDatabase):
    """A wrapper around multiple databases.

    Each ``save``, ``fetch``, ``move``, or ``delete`` operation will be run against
    all of the wrapped databases.  ``fetch`` does not yield duplicate values, even
    if the same value is present in two or more of the wrapped databases.

    This combines well with a :class:`ReadOnlyDatabase`, as follows:

    .. code-block:: python

        local = DirectoryBasedExampleDatabase("/tmp/hypothesis/examples/")
        shared = CustomNetworkDatabase()

        settings.register_profile("ci", database=shared)
        settings.register_profile(
            "dev", database=MultiplexedDatabase(local, ReadOnlyDatabase(shared))
        )
        settings.load_profile("ci" if os.environ.get("CI") else "dev")

    So your CI system or fuzzing runs can populate a central shared database;
    while local runs on development machines can reproduce any failures from CI
    but will only cache their own failures locally and cannot remove examples
    from the shared database.
    """

    def __init__(self, *dbs: ExampleDatabase) -> None:
        super().__init__()
        assert all(isinstance(db, ExampleDatabase) for db in dbs)
        self._wrapped = dbs

    def __repr__(self) -> str:
        return "MultiplexedDatabase({})".format(", ".join(map(repr, self._wrapped)))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, MultiplexedDatabase) and self._wrapped == other._wrapped
        )

    def fetch(self, key: bytes) -> Iterable[bytes]:
        seen = set()
        for db in self._wrapped:
            for value in db.fetch(key):
                if value not in seen:
                    yield value
                    seen.add(value)

    def save(self, key: bytes, value: bytes) -> None:
        for db in self._wrapped:
            db.save(key, value)

    def delete(self, key: bytes, value: bytes) -> None:
        for db in self._wrapped:
            db.delete(key, value)

    def move(self, src: bytes, dest: bytes, value: bytes) -> None:
        for db in self._wrapped:
            db.move(src, dest, value)

    @property
    def capabilities(self) -> frozenset[str]:
        caps = [db.capabilities for db in self._wrapped]
        return frozenset.intersection(*caps) if caps else frozenset()

    def read_many(self, ops: Sequence[ReadOpT]) -> list[Any]:
        ops = list(ops)
        per_db = [db.read_many(ops) for db in self._wrapped]
        results: list[Any] = []
        for i, op in enumerate(ops):
            found = [r[i] for r in per_db]
            if isinstance(op, MapGet):
                results.append(next((v for v in found if v is not None), None))
            elif isinstance(op, MapItems):
                # Earlier databases take precedence.
                results.append(
                    {k: v for items in reversed(found) for k, v in items.items()}
                )
            else:
                merged = sorted({entry for entries in found for entry in entries})
                results.append(_select_entries(merged, op))
        return results

    def write_many(self, ops: Sequence[WriteOpT], *, atomic: bool = False) -> list[Any]:
        ops = list(ops)
        results = [db.write_many(ops, atomic=atomic) for db in self._wrapped]
        return results[0] if results else _not_applied(ops)

    def journal_head(self, partition: KeyPartT) -> bytes:
        return _pack_cursors([db.journal_head(partition) for db in self._wrapped])

    def journal_read(
        self,
        cursors: Mapping[KeyPartT, bytes],
        *,
        timeout: float | None = 0,
        limit: int | None = None,
    ) -> tuple[list[Change], dict[KeyPartT, bytes]]:
        per_db: list[dict[KeyPartT, bytes]] = [{} for _ in self._wrapped]
        for partition, cursor in cursors.items():
            parts = _unpack_cursors(cursor, len(self._wrapped))
            for i, part in enumerate(parts):
                per_db[i][partition] = part
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            changes: list[Change] = []
            for i, db in enumerate(self._wrapped):
                found, per_db[i] = db.journal_read(per_db[i], timeout=0, limit=limit)
                changes.extend(found)
            remaining = None if deadline is None else deadline - time.monotonic()
            if changes or (remaining is not None and remaining <= 0):
                new = {p: _pack_cursors([c[p] for c in per_db]) for p in cursors}
                return changes, new
            time.sleep(0.05 if remaining is None else min(0.05, remaining))

    def current_time(self) -> float:
        return self._wrapped[0].current_time() if self._wrapped else time.time()

    def flush(self, timeout: float | None = None) -> None:
        for db in self._wrapped:
            db.flush(timeout)

    def close(self) -> None:
        super().close()
        for db in self._wrapped:
            db.close()

    def _start_listening(self) -> None:
        for db in self._wrapped:
            db.add_listener(self._broadcast_change)

    def _stop_listening(self) -> None:
        for db in self._wrapped:
            db.remove_listener(self._broadcast_change)


class GitHubArtifactDatabase(ExampleDatabase):
    """
    A file-based database loaded from a `GitHub Actions <https://docs.github.com/en/actions>`_ artifact.

    You can use this for sharing example databases between CI runs and developers, allowing
    the latter to get read-only access to the former. This is particularly useful for
    continuous fuzzing (i.e. with `HypoFuzz <https://hypofuzz.com/>`_),
    where the CI system can help find new failing test cases through fuzzing,
    and developers can reproduce them locally without any manual effort.

    .. note::
        You must provide ``GITHUB_TOKEN`` as an environment variable. In CI, Github Actions provides
        this automatically, but it needs to be set manually for local usage. In a developer machine,
        this would usually be a `Personal Access Token <https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens>`_.
        If the repository is private, it's necessary for the token to have ``repo`` scope
        in the case of a classic token, or ``actions:read`` in the case of a fine-grained token.


    In most cases, this will be used
    through the :class:`~hypothesis.database.MultiplexedDatabase`,
    by combining a local directory-based database with this one. For example:

    .. code-block:: python

        local = DirectoryBasedExampleDatabase(".hypothesis/examples")
        shared = ReadOnlyDatabase(GitHubArtifactDatabase("user", "repo"))

        settings.register_profile("ci", database=local)
        settings.register_profile("dev", database=MultiplexedDatabase(local, shared))
        # We don't want to use the shared database in CI, only to populate its local one.
        # which the workflow should then upload as an artifact.
        settings.load_profile("ci" if os.environ.get("CI") else "dev")

    .. note::
        Because this database is read-only, you always need to wrap it with the
        :class:`ReadOnlyDatabase`.

    A setup like this can be paired with a GitHub Actions workflow including
    something like the following:

    .. code-block:: yaml

        - name: Download example database
          uses: dawidd6/action-download-artifact@v9
          with:
            name: hypothesis-example-db
            path: .hypothesis/examples
            if_no_artifact_found: warn
            workflow_conclusion: completed

        - name: Run tests
          run: pytest

        - name: Upload example database
          uses: actions/upload-artifact@v3
          if: always()
          with:
            name: hypothesis-example-db
            path: .hypothesis/examples

    In this workflow, we use `dawidd6/action-download-artifact <https://github.com/dawidd6/action-download-artifact>`_
    to download the latest artifact given that the official `actions/download-artifact <https://github.com/actions/download-artifact>`_
    does not support downloading artifacts from previous workflow runs.

    The database automatically implements a simple file-based cache with a default expiration period
    of 1 day. You can adjust this through the ``cache_timeout`` property.

    For mono-repo support, you can provide a unique ``artifact_name`` (e.g. ``hypofuzz-example-db-frontend``).
    """

    def __init__(
        self,
        owner: str,
        repo: str,
        artifact_name: str = "hypothesis-example-db",
        cache_timeout: timedelta = timedelta(days=1),
        path: StrPathT | None = None,
    ):
        super().__init__()
        self.owner = owner
        self.repo = repo
        self.artifact_name = artifact_name
        self.cache_timeout = cache_timeout

        # Get the GitHub token from the environment
        # It's unnecessary to use a token if the repo is public
        self.token: str | None = getenv("GITHUB_TOKEN")

        self._storage_dir: StorageDirectory | None = None
        if path is None:
            self._storage_dir = storage_directory(
                f"github-artifacts/{self.artifact_name}/"
            )
            self.path = self._storage_dir.path
        else:
            self.path = Path(path)

        # We don't want to initialize the cache until we need to
        self._initialized: bool = False
        self._disabled: bool = False

        # This is the path to the artifact in usage
        # .hypothesis/github-artifacts/<artifact-name>/<modified_isoformat>.zip
        self._artifact: Path | None = None
        # This caches the artifact structure
        self._access_cache: dict[PurePath, set[PurePath]] | None = None

        # Message to display if user doesn't wrap around ReadOnlyDatabase
        self._read_only_message = (
            "This database is read-only. "
            "Please wrap this class with ReadOnlyDatabase"
            "i.e. ReadOnlyDatabase(GitHubArtifactDatabase(...))."
        )

    def __repr__(self) -> str:
        return (
            f"GitHubArtifactDatabase(owner={self.owner!r}, "
            f"repo={self.repo!r}, artifact_name={self.artifact_name!r})"
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, GitHubArtifactDatabase)
            and self.owner == other.owner
            and self.repo == other.repo
            and self.artifact_name == other.artifact_name
            and self.path == other.path
        )

    def _prepare_for_io(self) -> None:
        assert self._artifact is not None, "Artifact not loaded."

        if self._initialized:  # pragma: no cover
            return

        # Test that the artifact is valid
        try:
            with ZipFile(self._artifact) as f:
                if f.testzip():  # pragma: no cover
                    raise BadZipFile

            # Turns out that testzip() doesn't work quite well
            # doing the cache initialization here instead
            # will give us more coverage of the artifact.

            # Cache the files inside each keypath
            self._access_cache = {}
            with ZipFile(self._artifact) as zf:
                namelist = zf.namelist()
                # Iterate over files in the artifact
                for filename in namelist:
                    fileinfo = zf.getinfo(filename)
                    if fileinfo.is_dir():
                        self._access_cache.setdefault(PurePath(filename), set())
                    else:
                        # Get the keypath from the filename
                        keypath = PurePath(filename).parent
                        # Add the file to the keypath
                        self._access_cache.setdefault(keypath, set()).add(
                            PurePath(filename)
                        )
        except BadZipFile:
            warnings.warn(
                "The downloaded artifact from GitHub is invalid. "
                "This could be because the artifact was corrupted, "
                "or because the artifact was not created by Hypothesis. ",
                HypothesisWarning,
                stacklevel=3,
            )
            self._disabled = True

        self._initialized = True

    def _initialize_db(self) -> None:
        # Trigger warning that we suppressed earlier by intent_to_write=False
        storage_directory(self.path.name)
        # Create the cache directory if it doesn't exist
        if self._storage_dir is not None:  # pragma: no cover
            self._storage_dir.create_if_missing()
        else:
            self.path.mkdir(exist_ok=True, parents=True)

        # Get all artifacts
        cached_artifacts = sorted(
            self.path.glob("*.zip"),
            key=lambda a: datetime.fromisoformat(a.stem.replace("_", ":")),
        )

        # Remove all but the latest artifact
        for artifact in cached_artifacts[:-1]:
            artifact.unlink()

        try:
            found_artifact = cached_artifacts[-1]
        except IndexError:
            found_artifact = None

        # Check if the latest artifact is a cache hit
        if found_artifact is not None and (
            datetime.now(timezone.utc)
            - datetime.fromisoformat(found_artifact.stem.replace("_", ":"))
            < self.cache_timeout
        ):
            self._artifact = found_artifact
        else:
            # Download the latest artifact from GitHub
            new_artifact = self._fetch_artifact()

            if new_artifact:
                if found_artifact is not None:
                    found_artifact.unlink()
                self._artifact = new_artifact
            elif found_artifact is not None:
                warnings.warn(
                    "Using an expired artifact as a fallback for the database: "
                    f"{found_artifact}",
                    HypothesisWarning,
                    stacklevel=2,
                )
                self._artifact = found_artifact
            else:
                warnings.warn(
                    "Couldn't acquire a new or existing artifact. Disabling database.",
                    HypothesisWarning,
                    stacklevel=2,
                )
                self._disabled = True
                return

        self._prepare_for_io()

    def _get_bytes(self, url: str) -> bytes | None:  # pragma: no cover
        request = Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28 ",
            },
        )
        # see https://github.com/HypothesisWorks/hypothesis/pull/4791
        request.add_unredirected_header("Authorization", f"Bearer {self.token}")
        warning_message = None
        response_bytes: bytes | None = None
        try:
            with urlopen(request) as response:
                response_bytes = response.read()
        except HTTPError as e:
            if e.code == 401:
                warning_message = (
                    "Authorization failed when trying to download artifact from GitHub. "
                    "Check that you have a valid GITHUB_TOKEN set in your environment."
                )
            else:
                warning_message = (
                    "Could not get the latest artifact from GitHub. "
                    "This could be because the repository "
                    "or artifact does not exist. "
                )
            # see https://github.com/python/cpython/issues/128734
            e.close()
        except URLError:
            warning_message = "Could not connect to GitHub to get the latest artifact. "
        except TimeoutError:
            warning_message = (
                "Could not connect to GitHub to get the latest artifact "
                "(connection timed out)."
            )

        if warning_message is not None:
            warnings.warn(warning_message, HypothesisWarning, stacklevel=4)
            return None

        return response_bytes

    def _fetch_artifact(self) -> Path | None:  # pragma: no cover
        # Get the list of artifacts from GitHub
        url = f"https://api.github.com/repos/{self.owner}/{self.repo}/actions/artifacts"
        response_bytes = self._get_bytes(url)
        if response_bytes is None:
            return None

        artifacts = json.loads(response_bytes)["artifacts"]
        artifacts = [a for a in artifacts if a["name"] == self.artifact_name]

        if not artifacts:
            return None

        # Get the latest artifact from the list
        artifact = max(artifacts, key=lambda a: a["created_at"])
        url = artifact["archive_download_url"]

        # Download the artifact
        artifact_bytes = self._get_bytes(url)
        if artifact_bytes is None:
            return None

        # Save the artifact to the cache
        # We replace ":" with "_" to ensure the filenames are compatible
        # with Windows filesystems
        timestamp = datetime.now(timezone.utc).isoformat().replace(":", "_")
        artifact_path = self.path / f"{timestamp}.zip"
        try:
            artifact_path.write_bytes(artifact_bytes)
        except OSError:
            warnings.warn(
                "Could not save the latest artifact from GitHub. ",
                HypothesisWarning,
                stacklevel=3,
            )
            return None

        return artifact_path

    @staticmethod
    @lru_cache
    def _key_path(key: bytes) -> PurePath:
        return PurePath(_hash(key) + "/")

    def fetch(self, key: bytes) -> Iterable[bytes]:
        if self._disabled:
            return

        if not self._initialized:
            self._initialize_db()
            if self._disabled:
                return

        assert self._artifact is not None
        assert self._access_cache is not None

        kp = self._key_path(key)

        with ZipFile(self._artifact) as zf:
            # Get all the files in the kp from the cache
            filenames = self._access_cache.get(kp, ())
            for filename in filenames:
                with zf.open(filename.as_posix()) as f:
                    yield f.read()

    # Read-only interface
    def save(self, key: bytes, value: bytes) -> None:
        raise RuntimeError(self._read_only_message)

    def move(self, src: bytes, dest: bytes, value: bytes) -> None:
        raise RuntimeError(self._read_only_message)

    def delete(self, key: bytes, value: bytes) -> None:
        raise RuntimeError(self._read_only_message)


class BackgroundWriteDatabase(ExampleDatabase):
    """A wrapper which defers writes on the given database to a background thread.

    Calls to :meth:`~hypothesis.database.ExampleDatabase.fetch` wait for any
    enqueued writes to finish before fetching from the database.
    """

    def __init__(self, db: ExampleDatabase) -> None:
        super().__init__()
        self._db = db
        self._queue: Queue[tuple[str, tuple[Any, ...]]] = Queue()
        self._thread: Thread | None = None

    def _ensure_thread(self):
        if self._thread is None:
            self._thread = Thread(target=self._worker, daemon=True)
            self._thread.start()
            # avoid an unbounded timeout during gc. 0.1 should be plenty for most
            # use cases.
            weakref.finalize(self, self._join, 0.1)

    def __repr__(self) -> str:
        return f"BackgroundWriteDatabase({self._db!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BackgroundWriteDatabase) and self._db == other._db

    def _worker(self) -> None:
        while True:
            method, args = self._queue.get()
            if not method:  # close() asks the thread to stop
                self._queue.task_done()
                return
            try:
                getattr(self._db, method)(*args)
            except Exception as err:
                warnings.warn(
                    f"{self!r} could not apply a queued {method}: {err!r}",
                    HypothesisWarning,
                    stacklevel=1,
                )
            finally:
                self._queue.task_done()

    def _join(self, timeout: float | None = None) -> None:
        # copy of Queue.join with a timeout. https://bugs.python.org/issue9634
        with self._queue.all_tasks_done:
            while self._queue.unfinished_tasks:
                self._queue.all_tasks_done.wait(timeout)

    def fetch(self, key: bytes) -> Iterable[bytes]:
        self._join()
        return self._db.fetch(key)

    def save(self, key: bytes, value: bytes) -> None:
        self._ensure_thread()
        self._queue.put(("save", (key, value)))

    def delete(self, key: bytes, value: bytes) -> None:
        self._ensure_thread()
        self._queue.put(("delete", (key, value)))

    def move(self, src: bytes, dest: bytes, value: bytes) -> None:
        self._ensure_thread()
        self._queue.put(("move", (src, dest, value)))

    @property
    def capabilities(self) -> frozenset[str]:
        return self._db.capabilities

    def read_many(self, ops: Sequence[ReadOpT]) -> list[Any]:
        self._join()
        return self._db.read_many(ops)

    def write_many(self, ops: Sequence[WriteOpT], *, atomic: bool = False) -> list[Any]:
        ops = list(ops)
        if atomic or any(_is_conditional(op) for op in ops):
            self._join()
            return self._db.write_many(ops, atomic=atomic)
        self._ensure_thread()
        self._queue.put(("write_many", (ops,)))
        return [None] * len(ops)

    def journal_head(self, partition: KeyPartT) -> bytes:
        return self._db.journal_head(partition)

    def journal_read(
        self,
        cursors: Mapping[KeyPartT, bytes],
        *,
        timeout: float | None = 0,
        limit: int | None = None,
    ) -> tuple[list[Change], dict[KeyPartT, bytes]]:
        return self._db.journal_read(cursors, timeout=timeout, limit=limit)

    def current_time(self) -> float:
        return self._db.current_time()

    def close(self) -> None:
        self._join()
        if self._thread is not None:
            self._queue.put(("", ()))
            self._thread.join()
            self._thread = None
        super().close()
        self._db.close()

    def flush(self, timeout: float | None = None) -> None:
        self._join(timeout)
        self._db.flush(timeout)

    def _start_listening(self) -> None:
        self._db.add_listener(self._broadcast_change)

    def _stop_listening(self) -> None:
        self._db.remove_listener(self._broadcast_change)


_PUT, _DELETE, _CLEAR, _APPEND, _INVALIDATE = 1, 2, 3, 4, 5
_CHANGE_OPS: dict[int, ChangeOpT] = {
    _PUT: "put",
    _DELETE: "delete",
    _CLEAR: "clear",
    _APPEND: "append",
    _INVALIDATE: "invalidate",
}
#: Journal entries include values up to this size. Larger values arrive as None.
INLINE_VALUE_LIMIT = 64 * 1024


def _journal_change(
    op: int,
    key: bytes,
    field: bytes | None,
    entry_id: bytes | None,
    value: bytes | None,
) -> Change:
    """Build a Change from a journal row, as stored by the SQL backends."""
    return Change(
        _CHANGE_OPS[op],
        decode(key),
        None if field is None else decode(field),
        entry_id,
        value,
    )


_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS maps (
    kh BLOB NOT NULL, fh BLOB NOT NULL, key BLOB NOT NULL, field BLOB NOT NULL,
    value BLOB NOT NULL, exp REAL, PRIMARY KEY (kh, fh)
);
CREATE INDEX IF NOT EXISTS maps_exp ON maps (exp) WHERE exp IS NOT NULL;
CREATE TABLE IF NOT EXISTS logs (
    kh BLOB NOT NULL, id BLOB NOT NULL, value BLOB NOT NULL, PRIMARY KEY (kh, id)
);
CREATE TABLE IF NOT EXISTS log_meta (
    kh BLOB PRIMARY KEY, key BLOB NOT NULL, last_id BLOB NOT NULL,
    count INTEGER NOT NULL, ttl REAL
);
CREATE TABLE IF NOT EXISTS journal (
    id INTEGER PRIMARY KEY AUTOINCREMENT, at REAL NOT NULL, ph BLOB NOT NULL,
    op INTEGER NOT NULL, key BLOB NOT NULL, field BLOB, eid BLOB, value BLOB
);
CREATE INDEX IF NOT EXISTS journal_at ON journal (at);
"""


class _Connections:
    """One connection for each thread, opened when the thread first asks.

    A forked process opens its own connections, because it must not use its
    parent's. ``close`` closes every connection that this process opened.
    """

    def __init__(self, connect: Callable[[], Any]) -> None:
        self._connect = connect
        self._local = threading.local()
        self._lock = threading.Lock()
        self._opened: list[Any] = []
        self._pid = os.getpid()

    def get(self) -> Any:
        conn = getattr(self._local, "conn", None)
        # psycopg connections say when they have been closed, as by a restart.
        if (
            conn is None
            or getattr(conn, "closed", False)
            or self._local.pid != os.getpid()
        ):
            conn = self._connect()
            self._local.conn, self._local.pid = conn, os.getpid()
            with self._lock:
                if self._pid != os.getpid():
                    self._opened, self._pid = [], os.getpid()
                self._opened.append(conn)
        return conn

    def close(self) -> None:
        with self._lock:
            opened = self._opened if self._pid == os.getpid() else []
            self._opened, self._pid = [], os.getpid()
            self._local = threading.local()
        for conn in opened:
            conn.close()


class _SQLiteListener(_ListenerThread):
    def __init__(self, db: "SQLiteExampleDatabase") -> None:
        super().__init__(db)
        self.conn = db._open()
        self.position = db._head(self.conn)
        self.poll_interval = db._poll_interval

    def fetch(self) -> list[Change]:
        rows = self.conn.execute(
            "SELECT id, op, key, field, eid, value FROM journal WHERE id > ? "
            "ORDER BY id LIMIT 1000",
            (self.position,),
        ).fetchall()
        if rows:
            self.position = rows[-1][0]
        return [_journal_change(*row[1:]) for row in rows]

    def wait(self) -> None:
        self.stopping.wait(self.poll_interval)

    def release(self) -> None:
        self.conn.close()


class SQLiteExampleDatabase(_NativeDatabase):
    """Store examples in a SQLite database file.

    Any number of processes on one machine can use the same file. The file must
    not be on a network filesystem, because SQLite's write-ahead log needs
    shared memory.
    """

    _listener_thread_class = _SQLiteListener

    def __init__(
        self,
        path: StrPathT,
        *,
        journal_retention: float = 300.0,
        poll_interval: float = 0.01,
    ) -> None:
        super().__init__()
        self.path = Path(path)
        self.journal_retention = journal_retention
        self._poll_interval = poll_interval
        self._connections = _Connections(self._open)
        self._next_cleanup = time.time() + 1

    def __repr__(self) -> str:
        return f"SQLiteExampleDatabase({self.path!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SQLiteExampleDatabase) and self.path == other.path

    def __getstate__(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "journal_retention": self.journal_retention,
            "poll_interval": self._poll_interval,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(**state)  # type: ignore

    @property
    def capabilities(self) -> frozenset[str]:
        return frozenset({"native", "atomic", "journal", "blocking", "ttl"})

    def _open(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            self.path, timeout=60, isolation_level=None, check_same_thread=False
        )
        self._use_wal(conn)
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(_SQLITE_SCHEMA)
        return conn

    def _conn(self) -> sqlite3.Connection:
        return self._connections.get()

    def close(self) -> None:
        super().close()
        self._connections.close()

    @staticmethod
    def _use_wal(conn: sqlite3.Connection) -> None:
        # The journal mode is stored in the file, so check it before setting it.
        # Setting it takes a lock, and SQLite does not wait for that lock, so
        # processes that open a new file together can fail here. Retry them.
        for attempt in range(50):
            try:
                if conn.execute("PRAGMA journal_mode").fetchone()[0] != "wal":
                    conn.execute("PRAGMA journal_mode=WAL")
                return
            except sqlite3.OperationalError:
                if attempt == 49:
                    raise
                time.sleep(0.1)

    @staticmethod
    def _partition_hash(key: KeyTupleT) -> bytes:
        return short_hash(encode(key[:1]))

    @staticmethod
    def _inline(value: bytes) -> bytes | None:
        return value if len(value) <= INLINE_VALUE_LIMIT else None

    def read_many(self, ops: Sequence[ReadOpT]) -> list[Any]:
        conn = self._conn()
        now = time.time()
        results: list[Any] = []
        for op in ops:
            ek = encode(op.key)
            kh = short_hash(ek)
            if isinstance(op, MapGet):
                ef = encode(op.field)
                row = conn.execute(
                    "SELECT value, exp FROM maps WHERE kh=? AND fh=? AND key=? AND field=?",
                    (kh, short_hash(ef), ek, ef),
                ).fetchone()
                results.append(
                    row[0] if row and (row[1] is None or row[1] > now) else None
                )
            elif isinstance(op, MapItems):
                prefix = encode(op.prefix)
                rows = conn.execute(
                    "SELECT field, value, exp FROM maps WHERE kh=? AND key=?", (kh, ek)
                )
                results.append(
                    {
                        decode(field): value
                        for field, value, exp in rows
                        if (exp is None or exp > now) and field.startswith(prefix)
                    }
                )
            else:
                sql = "SELECT id, value FROM logs WHERE kh=?"
                params: list[Any] = [kh]
                if op.after is not None:
                    sql += " AND id > ?"
                    params.append(op.after)
                if op.before is not None:
                    sql += " AND id < ?"
                    params.append(op.before)
                sql += " ORDER BY id DESC" if op.reverse else " ORDER BY id"
                if op.limit is not None:
                    sql += " LIMIT ?"
                    params.append(op.limit)
                results.append([tuple(row) for row in conn.execute(sql, params)])
        return results

    def write_many(self, ops: Sequence[WriteOpT], *, atomic: bool = False) -> list[Any]:
        ops = list(ops)
        if not ops:
            return []
        if atomic:
            _check_atomic(ops)
        conn = self._conn()
        now = time.time()
        journal: list[tuple[Any, ...]] = []
        conn.execute("BEGIN IMMEDIATE")
        try:
            if atomic and not all(
                self._current(conn, op, now) == op.expect
                for op in ops
                if _is_conditional(op)
            ):
                conn.execute("ROLLBACK")
                return _not_applied(ops)
            results = [
                self._write(conn, op, now, journal, check=not atomic) for op in ops
            ]
            if journal:
                conn.executemany(
                    "INSERT INTO journal (at, ph, op, key, field, eid, value) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    journal,
                )
            conn.execute("COMMIT")
        finally:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
        if now > self._next_cleanup:
            self._cleanup(conn, now)
        return results

    @staticmethod
    def _row(
        conn: sqlite3.Connection, ek: bytes, ef: bytes
    ) -> tuple[bytes, float | None] | None:
        return conn.execute(
            "SELECT value, exp FROM maps WHERE kh=? AND fh=? AND key=? AND field=?",
            (short_hash(ek), short_hash(ef), ek, ef),
        ).fetchone()

    def _current(
        self, conn: sqlite3.Connection, op: MapPut | MapDelete, now: float
    ) -> bytes | None:
        row = self._row(conn, encode(op.key), encode(op.field))
        return row[0] if row and (row[1] is None or row[1] > now) else None

    def _write(
        self,
        conn: sqlite3.Connection,
        op: WriteOpT,
        now: float,
        journal: list[tuple[Any, ...]],
        *,
        check: bool,
    ) -> Any:
        ek = encode(op.key)
        kh = short_hash(ek)
        ph = self._partition_hash(op.key)
        if isinstance(op, (MapPut, MapDelete)):
            ef = encode(op.field)
            fh = short_hash(ef)
            row = self._row(conn, ek, ef)
            current = row[0] if row and (row[1] is None or row[1] > now) else None
            if check and _is_conditional(op) and current != op.expect:
                return False
            if isinstance(op, MapPut):
                expires = None if op.ttl is None else now + op.ttl
                if (
                    current != op.value
                    or expires is not None
                    or (row and row[1] is not None)
                ):
                    conn.execute(
                        "INSERT OR REPLACE INTO maps VALUES (?, ?, ?, ?, ?, ?)",
                        (kh, fh, ek, ef, op.value, expires),
                    )
                if current != op.value:
                    journal.append(
                        (now, ph, _PUT, ek, ef, None, self._inline(op.value))
                    )
                return True
            conn.execute(
                "DELETE FROM maps WHERE kh=? AND fh=? AND key=? AND field=?",
                (kh, fh, ek, ef),
            )
            if current is None:
                return False
            journal.append((now, ph, _DELETE, ek, ef, None, None))
            return True
        if isinstance(op, MapClear):
            if conn.execute("DELETE FROM maps WHERE kh=? AND key=?", (kh, ek)).rowcount:
                journal.append((now, ph, _CLEAR, ek, None, None, None))
            return None
        if isinstance(op, LogTrim):
            removed = 0
            for entry_id in op.ids:
                row = conn.execute(
                    "SELECT value FROM logs WHERE kh=? AND id=?", (kh, entry_id)
                ).fetchone()
                if row:
                    conn.execute("DELETE FROM logs WHERE kh=? AND id=?", (kh, entry_id))
                    removed += 1
                    value = self._inline(row[0])
                    journal.append((now, ph, _DELETE, ek, None, entry_id, value))
            removed += self._trim(conn, kh, maxlen=op.maxlen, before=op.before)
            conn.execute(
                "UPDATE log_meta SET count = count - ? WHERE kh=?", (removed, kh)
            )
            return removed
        meta = conn.execute(
            "SELECT last_id, count FROM log_meta WHERE kh=?", (kh,)
        ).fetchone()
        entry_id = next_entry_id(meta[0] if meta else None, int(now * 1000))
        count = (meta[1] if meta else 0) + 1
        conn.execute("INSERT INTO logs VALUES (?, ?, ?)", (kh, entry_id, op.value))
        if op.maxlen is not None and count > op.maxlen + max(1, op.maxlen // 4):
            count -= self._trim(conn, kh, maxlen=op.maxlen)
        conn.execute(
            "INSERT OR REPLACE INTO log_meta VALUES (?, ?, ?, ?, ?)",
            (kh, ek, entry_id, count, op.ttl),
        )
        journal.append((now, ph, _APPEND, ek, None, entry_id, self._inline(op.value)))
        return entry_id

    @staticmethod
    def _trim(
        conn: sqlite3.Connection,
        kh: bytes,
        *,
        maxlen: int | None = None,
        before: bytes | None = None,
    ) -> int:
        removed = 0
        if before is not None:
            removed += conn.execute(
                "DELETE FROM logs WHERE kh=? AND id<?", (kh, before)
            ).rowcount
        if maxlen is not None:
            row = conn.execute(
                "SELECT id FROM logs WHERE kh=? ORDER BY id DESC LIMIT 1 OFFSET ?",
                (kh, maxlen),
            ).fetchone()
            if row:
                removed += conn.execute(
                    "DELETE FROM logs WHERE kh=? AND id<=?", (kh, row[0])
                ).rowcount
        return removed

    def _cleanup(self, conn: sqlite3.Connection, now: float) -> None:
        self._next_cleanup = now + min(10.0, self.journal_retention / 4)
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                "DELETE FROM journal WHERE at < ?", (now - self.journal_retention,)
            )
            conn.execute("DELETE FROM maps WHERE exp < ?", (now,))
            for kh, ttl in conn.execute(
                "SELECT kh, ttl FROM log_meta WHERE ttl IS NOT NULL"
            ).fetchall():
                cutoff = make_entry_id(max(0, int((now - ttl) * 1000)), 0)
                if removed := self._trim(conn, kh, before=cutoff):
                    conn.execute(
                        "UPDATE log_meta SET count = count - ? WHERE kh=?",
                        (removed, kh),
                    )
            conn.execute("COMMIT")
        finally:
            if conn.in_transaction:
                conn.execute("ROLLBACK")

    @staticmethod
    def _head(conn: sqlite3.Connection) -> int:
        # Ids are assigned in commit order, since SQLite has one writer at a time.
        row = conn.execute(
            "SELECT seq FROM sqlite_sequence WHERE name='journal'"
        ).fetchone()
        return row[0] if row else 0

    def _journal_position(self, partition: KeyPartT) -> bytes:
        return struct.pack(">q", self._head(self._conn()))

    def _journal_fetch(
        self, positions: Mapping[KeyPartT, bytes], limit: int | None
    ) -> _JournalBatch:
        ids = {p: _unpack_position(">q", pos)[0] for p, pos in positions.items()}
        by_hash = {self._partition_hash((p,)): p for p in ids}
        placeholders = ",".join("?" * len(by_hash))
        conn = self._conn()
        # Read the head, the rows, and data_version from one snapshot.
        conn.execute("BEGIN")
        try:
            head = self._head(conn)
            rows = conn.execute(
                "SELECT id, ph, op, key, field, eid, value FROM journal "
                f"WHERE id > ? AND ph IN ({placeholders}) ORDER BY id LIMIT ?",
                (min(ids.values()), *by_hash, -1 if limit is None else limit),
            ).fetchall()
            version = conn.execute("PRAGMA data_version").fetchone()[0]
        finally:
            conn.execute("COMMIT")
        changes = [
            _journal_change(op, key, field, eid, value)
            for id_, ph, op, key, field, eid, value in rows
            if id_ > ids[by_hash[ph]]
        ]
        # Ids are shared by every partition, so a read that the limit cut short
        # may have cut short any of them.
        truncated = limit is not None and len(rows) == limit
        upto = rows[-1][0] if truncated else head
        return _JournalBatch(
            changes,
            {p: struct.pack(">q", max(id_, upto)) for p, id_ in ids.items()},
            set(ids) if truncated else set(),
            version,
        )

    def _journal_wait(self, batch: _JournalBatch, timeout: float | None) -> None:
        # Commits from other connections change data_version, which is cheap to poll.
        conn = self._conn()
        deadline = None if timeout is None else time.monotonic() + timeout
        while conn.execute("PRAGMA data_version").fetchone()[0] == batch.token:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return
            time.sleep(
                self._poll_interval
                if remaining is None
                else min(self._poll_interval, remaining)
            )


class ReadThroughDatabase(ExampleDatabase):
    """Copies data from a ``fallback`` database into ``primary`` as it is read.

    The first time a set is read, its members in ``fallback`` are copied into
    ``primary``, and a marker in ``primary`` records that this happened. Later
    reads and all writes use only ``primary``. Use this to move examples from an
    old database to a new one, without a separate migration step.
    """

    def __init__(self, primary: ExampleDatabase, fallback: ExampleDatabase) -> None:
        super().__init__()
        self.primary = primary
        self.fallback = fallback
        self._checked: set[bytes] = set()

    def __repr__(self) -> str:
        return f"ReadThroughDatabase({self.primary!r}, {self.fallback!r})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ReadThroughDatabase)
            and self.primary == other.primary
            and self.fallback == other.fallback
        )

    @property
    def capabilities(self) -> frozenset[str]:
        return self.primary.capabilities

    def _copy_up(self, keys: Iterable[bytes]) -> None:
        todo = list(dict.fromkeys(k for k in keys if k not in self._checked))
        if not todo:
            return
        markers = self.primary.read_many(
            [MapGet((k, "_meta"), ("read-through",)) for k in todo]
        )
        writes: list[WriteOpT] = []
        stamp = struct.pack(">d", time.time())
        for key, marker in zip(todo, markers, strict=True):
            if marker is None:
                writes.extend(
                    MapPut((key,), (value,)) for value in self.fallback.fetch(key)
                )
                writes.append(MapPut((key, "_meta"), ("read-through",), stamp))
        if writes:
            self.primary.write_many(writes)
        self._checked.update(todo)

    def fetch(self, key: bytes) -> Iterable[bytes]:
        self._copy_up([key])
        return self.primary.fetch(key)

    def save(self, key: bytes, value: bytes) -> None:
        self.primary.save(key, value)

    def delete(self, key: bytes, value: bytes) -> None:
        # Copy first, so that a later read cannot bring the value back.
        self._copy_up([key])
        self.primary.delete(key, value)

    def move(self, src: bytes, dest: bytes, value: bytes) -> None:
        self._copy_up([src, dest])
        self.primary.move(src, dest, value)

    def read_many(self, ops: Sequence[ReadOpT]) -> list[Any]:
        ops = list(ops)
        self._copy_up(cast(bytes, op.key[0]) for op in ops if is_legacy(op.key))
        return self.primary.read_many(ops)

    def write_many(self, ops: Sequence[WriteOpT], *, atomic: bool = False) -> list[Any]:
        ops = list(ops)
        self._copy_up(
            cast(bytes, op.key[0])
            for op in ops
            if is_legacy(op.key) and not isinstance(op, MapPut)
        )
        return self.primary.write_many(ops, atomic=atomic)

    def journal_head(self, partition: KeyPartT) -> bytes:
        return self.primary.journal_head(partition)

    def journal_read(
        self,
        cursors: Mapping[KeyPartT, bytes],
        *,
        timeout: float | None = 0,
        limit: int | None = None,
    ) -> tuple[list[Change], dict[KeyPartT, bytes]]:
        return self.primary.journal_read(cursors, timeout=timeout, limit=limit)

    def current_time(self) -> float:
        return self.primary.current_time()

    def flush(self, timeout: float | None = None) -> None:
        self.primary.flush(timeout)

    def close(self) -> None:
        super().close()
        self.primary.close()
        self.fallback.close()

    def _start_listening(self) -> None:
        self.primary.add_listener(self._broadcast_change)

    def _stop_listening(self) -> None:
        self.primary.remove_listener(self._broadcast_change)


class RemoteDatabase(_NativeDatabase):
    """A database served by another process on this machine.

    See :func:`serve_database`. Writes that are neither atomic nor conditional
    are queued, and return ``None``. Writes from one process are applied in
    order, and a read waits until that process's earlier writes have been
    applied. Instances can be pickled, so they can be passed to subprocesses.
    """

    #: Queued writes are sent together, after this many seconds or operations.
    batch_delay: float = 0.005
    batch_size: int = 200

    def __init__(self, address: Any, authkey: bytes) -> None:
        super().__init__()
        self.address = address
        self._authkey = authkey
        self._locks = {False: threading.Lock(), True: threading.Lock()}
        self._conns: dict[bool, Any] = {}
        self._pid = os.getpid()
        self._capabilities: frozenset[str] | None = None
        self._pending: list[WriteOpT] = []
        self._pending_lock = threading.Condition()
        self._flusher: threading.Thread | None = None

    def __repr__(self) -> str:
        return f"RemoteDatabase({self.address!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RemoteDatabase) and self.address == other.address

    def __reduce__(self) -> tuple[Any, ...]:
        return (RemoteDatabase, (self.address, self._authkey))

    def _check_pid(self) -> None:
        # After a fork, this process has no connections, no queued writes, and
        # no flusher thread. The caller holds _pending_lock.
        if self._pid != os.getpid():
            self._conns.clear()
            self._pending, self._flusher = [], None
            self._pid = os.getpid()

    def _flush_loop(self) -> None:
        # One thread per process. Starting a thread for each batch cost more than
        # sending the batch did. The thread stops when close() replaces it.
        me = threading.current_thread()
        while True:
            with self._pending_lock:
                while not self._pending and self._flusher is me:
                    self._pending_lock.wait()
                if self._flusher is not me:
                    return
            time.sleep(self.batch_delay)  # let more writes join the batch
            self._send_pending_later()

    def _send_pending(self) -> None:
        with self._locks[False]:
            self._send_pending_locked()

    def _send_pending_later(self) -> None:
        try:
            self._send_pending()
        except (OSError, EOFError) as err:
            warnings.warn(
                f"{self!r} could not send queued writes: {err!r}",
                HypothesisWarning,
                stacklevel=1,
            )

    def _take_pending(self) -> list[WriteOpT]:
        with self._pending_lock:
            self._check_pid()
            ops, self._pending = self._pending, []
        return ops

    def _send_pending_locked(self) -> None:
        # The caller holds the connection's lock, so writes are sent in order.
        if ops := self._take_pending():
            self._connection(journal=False).send((None, (), ops))

    def _connection(self, *, journal: bool) -> Any:
        conn = self._conns.get(journal)
        if conn is None:
            conn = self._conns[journal] = Client(self.address, authkey=self._authkey)
        return conn

    def _call(self, method: str, *args: Any, journal: bool = False) -> Any:
        # Each message is (method, args, queued writes). The server applies the
        # writes first, so this process's reads see its writes. Journal reads
        # block, so they use their own connection, and carry no writes.
        with self._locks[journal]:
            writes = [] if journal else self._take_pending()
            conn = self._connection(journal=journal)
            conn.send((method, args, writes))
            ok, result = conn.recv()
        if not ok:
            raise result
        return result

    @property
    def capabilities(self) -> frozenset[str]:
        if self._capabilities is None:
            self._capabilities = self._call("capabilities")
        return self._capabilities

    def read_many(self, ops: Sequence[ReadOpT]) -> list[Any]:
        return self._call("read_many", list(ops))

    def write_many(self, ops: Sequence[WriteOpT], *, atomic: bool = False) -> list[Any]:
        ops = list(ops)
        if atomic or any(_is_conditional(op) for op in ops):
            return self._call("write_many", ops, atomic)
        with self._pending_lock:
            self._check_pid()
            first = not self._pending
            self._pending.extend(ops)
            full = len(self._pending) >= self.batch_size
            if self._flusher is None:
                self._flusher = threading.Thread(
                    target=self._flush_loop, daemon=True, name="hypothesis-db-flush"
                )
                self._flusher.start()
            if first:
                self._pending_lock.notify()
        if full:
            self._send_pending()
        return [None] * len(ops)

    def journal_head(self, partition: KeyPartT) -> bytes:
        return self._call("journal_head", partition)

    def journal_read(
        self,
        cursors: Mapping[KeyPartT, bytes],
        *,
        timeout: float | None = 0,
        limit: int | None = None,
    ) -> tuple[list[Change], dict[KeyPartT, bytes]]:
        return self._call("journal_read", dict(cursors), timeout, limit, journal=True)

    def current_time(self) -> float:
        return self._call("current_time")

    def flush(self, timeout: float | None = None) -> None:
        self._call("flush", timeout)

    def close(self) -> None:
        super().close()
        self._send_pending_later()
        with self._pending_lock:
            self._flusher = None
            self._pending_lock.notify_all()
        for conn in self._conns.values():
            conn.close()
        self._conns.clear()


def serve_database(
    db: ExampleDatabase, *, address: Any = None, authkey: bytes | None = None
) -> "DatabaseServer":
    """Serve ``db`` to other processes on this machine, from a background thread.

    Returns a server whose ``client()`` method returns a :class:`RemoteDatabase`.
    Pass that to worker processes, so that they share this process's connection
    to the underlying store.
    """
    from hypothesis.internal.dbserver import DatabaseServer

    return DatabaseServer(db, address=address, authkey=authkey)


def choices_to_bytes(choices: Iterable[ChoiceT], /) -> bytes:
    """Serialize a list of choices to a bytestring.  Inverts choices_from_bytes."""
    # We use a custom serialization format for this, which might seem crazy - but our
    # data is a flat sequence of elements, and standard tools like protobuf or msgpack
    # don't deal well with e.g. nonstandard bit-pattern-NaNs, or invalid-utf8 unicode.
    #
    # We simply encode each element with a metadata byte, if needed a uint16 size, and
    # then the payload bytes.  For booleans, the payload is inlined into the metadata.
    parts = []
    for choice in choices:
        if isinstance(choice, bool):
            # `000_0000v` - tag zero, low bit payload.
            parts.append(b"\1" if choice else b"\0")
            continue

        # `tag_ssss [uint16 size?] [payload]`
        if isinstance(choice, float):
            tag = 1 << 5
            choice = struct.pack("!d", choice)
        elif isinstance(choice, int):
            tag = 2 << 5
            choice = choice.to_bytes(1 + choice.bit_length() // 8, "big", signed=True)
        elif isinstance(choice, bytes):
            tag = 3 << 5
        else:
            assert isinstance(choice, str)
            tag = 4 << 5
            choice = choice.encode(errors="surrogatepass")

        size = len(choice)
        if size < 0b11111:
            parts.append((tag | size).to_bytes(1, "big"))
        else:
            parts.append((tag | 0b11111).to_bytes(1, "big"))
            parts.append(_pack_uleb128(size))
        parts.append(choice)

    return b"".join(parts)


def _choices_from_bytes(buffer: bytes, /) -> tuple[ChoiceT, ...]:
    # See above for an explanation of the format.
    parts: list[ChoiceT] = []
    idx = 0
    while idx < len(buffer):
        tag = buffer[idx] >> 5
        size = buffer[idx] & 0b11111
        idx += 1

        if tag == 0:
            parts.append(bool(size))
            continue
        if size == 0b11111:
            offset, size = _unpack_uleb128(buffer[idx:])
            idx += offset
        chunk = buffer[idx : idx + size]
        idx += size

        if tag == 1:
            assert size == 8, "expected float64"
            parts.extend(struct.unpack("!d", chunk))
        elif tag == 2:
            parts.append(int.from_bytes(chunk, "big", signed=True))
        elif tag == 3:
            parts.append(chunk)
        else:
            assert tag == 4
            parts.append(chunk.decode(errors="surrogatepass"))
    return tuple(parts)


def choices_from_bytes(buffer: bytes, /) -> tuple[ChoiceT, ...] | None:
    """
    Deserialize a bytestring to a tuple of choices. Inverts choices_to_bytes.

    Returns None if the given bytestring is not a valid serialization of choice
    sequences.
    """
    try:
        return _choices_from_bytes(buffer)
    except Exception:
        # deserialization error, eg because our format changed or someone put junk
        # data in the db.
        return None
