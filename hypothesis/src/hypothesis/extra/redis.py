# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import base64
import json
import threading
import time
from collections.abc import Mapping, Sequence
from datetime import timedelta
from typing import Any

from redis import Redis, WatchError

from hypothesis.database import (
    INLINE_VALUE_LIMIT,
    Change,
    LogAppend,
    LogRange,
    LogTrim,
    MapClear,
    MapDelete,
    MapGet,
    MapPut,
    ReadOpT,
    WriteOpT,
    _check_atomic,
    _events_from_change,
    _is_conditional,
    _journal_change,
    _JournalBatch,
    _matches_prefix,
    _NativeDatabase,
    _not_applied,
    unset,
)
from hypothesis.internal.dbcodec import (
    KeyPartT,
    KeyTupleT,
    decode,
    encode,
    is_legacy,
    log_key,
    make_entry_id,
    split_entry_id,
)
from hypothesis.internal.validation import check_type

# Every write is one script, which applies the change, appends to the
# partition's journal stream, and publishes a wake-up. See section 5 of
# guides/database-design.md.

_PRELUDE = """
local function now_ms()
  local t = redis.call('TIME')
  return tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
end

local function journal(jkey, retention, wake, fields)
  redis.call('XADD', jkey, 'MINID', '~', now_ms() - tonumber(retention), '*', unpack(fields))
  redis.call('PUBLISH', wake, '')
end

local function firehose(channel, payload)
  if channel ~= '' then redis.call('PUBLISH', channel, payload) end
end

-- Remove expired fields, and expire the whole map when every field has a deadline.
local function expire_fields(map, exp, now)
  local deadlines = redis.call('HGETALL', exp)
  local latest = 0
  for i = 1, #deadlines, 2 do
    local deadline = tonumber(deadlines[i + 1])
    if deadline <= now then
      redis.call('HDEL', map, deadlines[i])
      redis.call('HDEL', exp, deadlines[i])
    elseif deadline > latest then
      latest = deadline
    end
  end
  local size = redis.call('HLEN', map)
  if size > 0 and redis.call('HLEN', exp) == size then
    redis.call('PEXPIRE', map, latest - now)
    redis.call('PEXPIRE', exp, latest - now)
  else
    redis.call('PERSIST', map)
    redis.call('PERSIST', exp)
  end
end

local function live_value(map, exp, field, now)
  local value = redis.call('HGET', map, field)
  if value then
    local deadline = redis.call('HGET', exp, field)
    if deadline and tonumber(deadline) <= now then return false end
  end
  return value
end
"""

# KEYS: set, journal. ARGV: member, expire_after, ek, ef, retention, wake,
# firehose channel, firehose payload, expect mode, expect value.
_SET_PUT = """
local present = redis.call('SISMEMBER', KEYS[1], ARGV[1]) == 1
if ARGV[9] == 'absent' and present then return 0 end
if ARGV[9] == 'eq' and (not present or ARGV[10] ~= '') then return 0 end
local added = redis.call('SADD', KEYS[1], ARGV[1])
redis.call('EXPIRE', KEYS[1], ARGV[2])
if added == 1 then
  journal(KEYS[2], ARGV[5], ARGV[6], {'o', '1', 'k', ARGV[3], 'f', ARGV[4], 'v', ''})
  firehose(ARGV[7], ARGV[8])
end
return 1
"""

_SET_DELETE = """
if ARGV[9] == 'eq' and ARGV[10] ~= '' then return 0 end
local removed = redis.call('SREM', KEYS[1], ARGV[1])
redis.call('EXPIRE', KEYS[1], ARGV[2])
if removed == 0 then return 0 end
journal(KEYS[2], ARGV[5], ARGV[6], {'o', '2', 'k', ARGV[3], 'f', ARGV[4]})
firehose(ARGV[7], ARGV[8])
return 1
"""

# KEYS: map or set, exp, journal. ARGV: ek, retention, wake, firehose channel, payload.
_CLEAR = """
local removed = redis.call('DEL', KEYS[1], KEYS[2])
if removed == 0 then return 0 end
journal(KEYS[3], ARGV[2], ARGV[3], {'o', '3', 'k', ARGV[1]})
firehose(ARGV[4], ARGV[5])
return 1
"""

# KEYS: map, exp, journal. ARGV: ef, value, ttl_ms, expect mode, expect value,
# ek, retention, inline limit, wake, firehose channel, firehose payload.
_MAP_PUT = """
local now = now_ms()
local current = live_value(KEYS[1], KEYS[2], ARGV[1], now)
if ARGV[4] == 'absent' and current then return 0 end
if ARGV[4] == 'eq' and current ~= ARGV[5] then return 0 end
redis.call('HSET', KEYS[1], ARGV[1], ARGV[2])
if ARGV[3] ~= '' then
  redis.call('HSET', KEYS[2], ARGV[1], now + tonumber(ARGV[3]))
else
  redis.call('HDEL', KEYS[2], ARGV[1])
end
expire_fields(KEYS[1], KEYS[2], now)
if current ~= ARGV[2] then
  local fields = {'o', '1', 'k', ARGV[6], 'f', ARGV[1]}
  if #ARGV[2] <= tonumber(ARGV[8]) then
    table.insert(fields, 'v')
    table.insert(fields, ARGV[2])
  end
  journal(KEYS[3], ARGV[7], ARGV[9], fields)
  firehose(ARGV[10], ARGV[11])
end
return 1
"""

# KEYS: map, exp, journal. ARGV: ef, expect mode, expect value, ek, retention,
# wake, firehose channel, firehose payload.
_MAP_DELETE = """
local now = now_ms()
local current = live_value(KEYS[1], KEYS[2], ARGV[1], now)
if ARGV[2] == 'eq' and current ~= ARGV[3] then return 0 end
redis.call('HDEL', KEYS[1], ARGV[1])
redis.call('HDEL', KEYS[2], ARGV[1])
expire_fields(KEYS[1], KEYS[2], now)
if not current then return 0 end
journal(KEYS[3], ARGV[5], ARGV[6], {'o', '2', 'k', ARGV[4], 'f', ARGV[1]})
firehose(ARGV[7], ARGV[8])
return 1
"""

# KEYS: stream, journal. ARGV: value, maxlen, ttl_ms, ek, retention, inline limit,
# wake, firehose channel, the log's reserved legacy key.
_LOG_APPEND = """
local id
if ARGV[2] ~= '' then
  id = redis.call('XADD', KEYS[1], 'MAXLEN', '~', ARGV[2], '*', 'v', ARGV[1])
else
  id = redis.call('XADD', KEYS[1], '*', 'v', ARGV[1])
end
if ARGV[3] ~= '' then
  redis.call('XTRIM', KEYS[1], 'MINID', '~', now_ms() - tonumber(ARGV[3]))
  redis.call('PEXPIRE', KEYS[1], ARGV[3])
else
  redis.call('PERSIST', KEYS[1])
end
local fields = {'o', '4', 'k', ARGV[4], 'i', id}
if #ARGV[1] <= tonumber(ARGV[6]) then
  table.insert(fields, 'v')
  table.insert(fields, ARGV[1])
end
journal(KEYS[2], ARGV[5], ARGV[7], fields)
-- Lua cannot encode base64, so this message is "A", then two length-prefixed
-- strings, then the value. A deleted entry's message starts with "D".
firehose(ARGV[8], 'A' .. #ARGV[9] .. ':' .. ARGV[9] .. #id .. ':' .. id .. ARGV[1])
return id
"""

# Reads filter out expired fields using the server's clock, as writes do.
# KEYS: map, exp. ARGV: field.
_MAP_GET = """
return live_value(KEYS[1], KEYS[2], ARGV[1], now_ms())
"""

# KEYS: map, exp. Returns a flat list of fields and values.
_MAP_ITEMS = """
local items = redis.call('HGETALL', KEYS[1])
if redis.call('EXISTS', KEYS[2]) == 0 then return items end
local now = now_ms()
local dead = {}
local deadlines = redis.call('HGETALL', KEYS[2])
for i = 1, #deadlines, 2 do
  if tonumber(deadlines[i + 1]) <= now then dead[deadlines[i]] = true end
end
local live = {}
for i = 1, #items, 2 do
  if not dead[items[i]] then
    table.insert(live, items[i])
    table.insert(live, items[i + 1])
  end
end
return live
"""

# KEYS: stream, journal. ARGV: maxlen, before, ek, retention, wake, firehose
# channel, legacy key, inline limit, then any ids to delete. Only those are journaled.
_LOG_TRIM = """
local removed = 0
for i = 9, #ARGV do
  local entry = redis.call('XRANGE', KEYS[1], ARGV[i], ARGV[i])[1]
  if entry then
    redis.call('XDEL', KEYS[1], ARGV[i])
    removed = removed + 1
    local value = entry[2][2]
    local fields = {'o', '2', 'k', ARGV[3], 'i', ARGV[i]}
    if #value <= tonumber(ARGV[8]) then
      table.insert(fields, 'v')
      table.insert(fields, value)
    end
    journal(KEYS[2], ARGV[4], ARGV[5], fields)
    firehose(ARGV[6], 'D' .. #ARGV[7] .. ':' .. ARGV[7] .. #ARGV[i] .. ':' .. ARGV[i] .. value)
  end
end
if ARGV[2] ~= '' then removed = removed + redis.call('XTRIM', KEYS[1], 'MINID', ARGV[2]) end
if ARGV[1] ~= '' then removed = removed + redis.call('XTRIM', KEYS[1], 'MAXLEN', ARGV[1]) end
return removed
"""


_MIN_ID = make_entry_id(0, 0)
_MAX_ID = make_entry_id(2**64 - 1, 2**64 - 1)


def _stream_id(entry_id: bytes) -> bytes:
    ms, seq = split_entry_id(entry_id)
    return b"%d-%d" % (ms, seq)


def _entry_id(stream_id: bytes | str) -> bytes:
    if isinstance(stream_id, str):
        stream_id = stream_id.encode()
    ms, seq = stream_id.split(b"-")
    return make_entry_id(int(ms), int(seq))


class RedisExampleDatabase(_NativeDatabase):
    """Store Hypothesis examples as sets in the given :class:`~redis.Redis` datastore.

    This is particularly useful for shared databases, as per the recipe
    for a :class:`~hypothesis.database.MultiplexedDatabase`.

    Maps are stored as hashes, and logs and journals as streams. Every key of a
    partition shares a hash tag. Requires Redis 6.2 or later.

    .. note::

        If a test has not been run for ``expire_after``, those examples will be allowed
        to expire.  The default time-to-live persists examples between weekly runs.
    """

    def __init__(
        self,
        redis: Redis,
        *,
        expire_after: timedelta = timedelta(days=8),
        key_prefix: bytes = b"hypothesis-example:",
        listener_channel: str = "hypothesis-changes",
        journal_retention: float = 300.0,
        sweep_interval: float = 5.0,
    ):
        super().__init__()
        check_type(Redis, redis, "redis")
        check_type(timedelta, expire_after, "expire_after")
        check_type(bytes, key_prefix, "key_prefix")
        check_type(str, listener_channel, "listener_channel")
        self.redis = redis
        self._expire_after = expire_after
        self._prefix = key_prefix
        self.listener_channel = listener_channel
        self.journal_retention = journal_retention
        self._sweep_interval = sweep_interval
        self._scripts = {
            name: redis.register_script(_PRELUDE + body)
            for name, body in [
                ("set_put", _SET_PUT),
                ("set_delete", _SET_DELETE),
                ("clear", _CLEAR),
                ("map_put", _MAP_PUT),
                ("map_delete", _MAP_DELETE),
                ("log_append", _LOG_APPEND),
                ("log_trim", _LOG_TRIM),
                ("map_get", _MAP_GET),
                ("map_items", _MAP_ITEMS),
            ]
        }
        self._pubsub: Any = None
        self._listen_thread: Any = None
        self._journal_lock = threading.Lock()
        self._journal_pubsub: Any = None
        self._wake_partitions: dict[bytes, KeyPartT] = {}
        self._dirty: set[KeyPartT] = set()
        self._read_upto: dict[KeyPartT, bytes] = {}
        self._last_swept: dict[KeyPartT, float] = {}

    def __repr__(self) -> str:
        return (
            f"RedisExampleDatabase({self.redis!r}, expire_after={self._expire_after!r})"
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, RedisExampleDatabase)
            and self.redis == other.redis
            and self._prefix == other._prefix
            and self.listener_channel == other.listener_channel
        )

    @property
    def capabilities(self) -> frozenset[str]:
        return frozenset(
            {"native", "atomic", "journal", "blocking", "shared", "ttl", "server_time"}
        )

    # Key layout

    def _tag(self, partition: KeyPartT) -> bytes:
        return (
            self._prefix
            + b"{"
            + base64.urlsafe_b64encode(encode((partition,))).rstrip(b"=")
            + b"}"
        )

    def _data_keys(self, key: KeyTupleT) -> tuple[bytes, bytes]:
        """The set or hash for ``key``, and the hash of its deadlines."""
        if is_legacy(key):
            return self._prefix + key[0], self._tag(key[0]) + b"e"  # type: ignore
        ek = encode(key)
        tag = self._tag(key[0])
        return tag + b"m" + ek, tag + b"e" + ek

    def _log_key(self, key: KeyTupleT) -> bytes:
        return self._tag(key[0]) + b"l" + encode(key)

    def _journal_key(self, partition: KeyPartT) -> bytes:
        return self._tag(partition) + b"j"

    def _wake_channel(self, partition: KeyPartT) -> bytes:
        return self._tag(partition) + b"w"

    def _firehose(self, change: Change) -> tuple[str, str]:
        if not self.listener_channel:
            return "", ""
        events = _events_from_change(change)
        if not events:
            return "", ""
        kind, (key, value) = events[0]
        payload = [
            kind,
            [self._encode(key), None if value is None else self._encode(value)],
        ]
        return self.listener_channel, json.dumps(payload)

    def _encode(self, value: bytes) -> str:
        return base64.b64encode(value).decode("ascii")

    def _decode(self, value: str) -> bytes:
        return base64.b64decode(value)

    # Reads

    def read_many(self, ops: Sequence[ReadOpT]) -> list[Any]:
        ops = list(ops)
        expire = int(self._expire_after.total_seconds())
        pipe = self.redis.pipeline(transaction=False)
        for op in ops:
            if isinstance(op, LogRange):
                if op.limit == 0 or op.before == _MIN_ID or op.after == _MAX_ID:
                    # Redis rejects exclusive bounds outside the range of ids.
                    pipe.echo(b"")
                    continue
                low = b"(" + _stream_id(op.after) if op.after is not None else b"-"
                high = b"(" + _stream_id(op.before) if op.before is not None else b"+"
                if op.reverse:
                    pipe.xrevrange(self._log_key(op.key), high, low, count=op.limit)
                else:
                    pipe.xrange(self._log_key(op.key), low, high, count=op.limit)
                continue
            data, exp = self._data_keys(op.key)
            if is_legacy(op.key):
                if isinstance(op, MapGet):
                    pipe.sismember(data, op.field[0])
                else:
                    pipe.smembers(data)
                pipe.expire(data, expire)
            elif isinstance(op, MapGet):
                self._scripts["map_get"](
                    keys=[data, exp], args=[encode(op.field)], client=pipe
                )
            else:
                self._scripts["map_items"](keys=[data, exp], client=pipe)
        replies = iter(pipe.execute())
        results: list[Any] = []
        for op in ops:
            reply = next(replies)
            if isinstance(op, LogRange):
                results.append(
                    [] if reply == b"" else [(_entry_id(i), f[b"v"]) for i, f in reply]
                )
                continue
            if is_legacy(op.key):
                next(replies)  # the reply to EXPIRE
                if isinstance(op, MapGet):
                    results.append(b"" if reply else None)
                else:
                    results.append(
                        {(m,): b"" for m in reply if _matches_prefix((m,), op.prefix)}
                    )
            elif isinstance(op, MapGet):
                results.append(reply)
            else:
                prefix = encode(op.prefix)
                pairs = zip(reply[::2], reply[1::2], strict=True)
                results.append(
                    {decode(ef): value for ef, value in pairs if ef.startswith(prefix)}
                )
        return results

    # Writes

    def _queue_write(self, pipe: Any, op: WriteOpT, *, check: bool) -> None:
        retention = int(self.journal_retention * 1000)
        wake = self._wake_channel(op.key[0])
        journal = self._journal_key(op.key[0])
        ek = encode(op.key)
        if isinstance(op, LogTrim):
            self._scripts["log_trim"](
                keys=[self._log_key(op.key), journal],
                args=[
                    "" if op.maxlen is None else op.maxlen,
                    "" if op.before is None else _stream_id(op.before),
                    ek,
                    retention,
                    wake,
                    self.listener_channel,
                    log_key(ek),
                    INLINE_VALUE_LIMIT,
                    *map(_stream_id, op.ids),
                ],
                client=pipe,
            )
            return
        if isinstance(op, LogAppend):
            self._scripts["log_append"](
                keys=[self._log_key(op.key), journal],
                args=[
                    op.value,
                    "" if op.maxlen is None else op.maxlen,
                    "" if op.ttl is None else int(op.ttl * 1000),
                    ek,
                    retention,
                    INLINE_VALUE_LIMIT,
                    wake,
                    self.listener_channel,
                    log_key(ek),
                ],
                client=pipe,
            )
            return
        data, exp = self._data_keys(op.key)
        if isinstance(op, MapClear):
            channel, payload = self._firehose(Change("clear", op.key))
            self._scripts["clear"](
                keys=[data, exp if not is_legacy(op.key) else data, journal],
                args=[ek, retention, wake, channel, payload],
                client=pipe,
            )
            return
        mode, expected = "", b""
        if check and op.expect is not unset:
            mode, expected = ("absent", b"") if op.expect is None else ("eq", op.expect)
        if isinstance(op, MapPut):
            channel, payload = self._firehose(
                Change("put", op.key, op.field, value=op.value)
            )
        else:
            channel, payload = self._firehose(Change("delete", op.key, op.field))
        if is_legacy(op.key):
            script = "set_put" if isinstance(op, MapPut) else "set_delete"
            self._scripts[script](
                keys=[data, journal],
                args=[
                    op.field[0],
                    int(self._expire_after.total_seconds()),
                    ek,
                    encode(op.field),
                    retention,
                    wake,
                    channel,
                    payload,
                    mode,
                    expected,
                ],
                client=pipe,
            )
        elif isinstance(op, MapPut):
            self._scripts["map_put"](
                keys=[data, exp, journal],
                args=[
                    encode(op.field),
                    op.value,
                    "" if op.ttl is None else int(op.ttl * 1000),
                    mode,
                    expected,
                    ek,
                    retention,
                    INLINE_VALUE_LIMIT,
                    wake,
                    channel,
                    payload,
                ],
                client=pipe,
            )
        else:
            self._scripts["map_delete"](
                keys=[data, exp, journal],
                args=[
                    encode(op.field),
                    mode,
                    expected,
                    ek,
                    retention,
                    wake,
                    channel,
                    payload,
                ],
                client=pipe,
            )

    @staticmethod
    def _result(op: WriteOpT, reply: Any) -> Any:
        if isinstance(op, (MapPut, MapDelete)):
            return bool(reply)
        if isinstance(op, LogAppend):
            return _entry_id(reply)
        if isinstance(op, LogTrim):
            return int(reply)
        return None

    def write_many(self, ops: Sequence[WriteOpT], *, atomic: bool = False) -> list[Any]:
        ops = list(ops)
        if not ops:
            return []
        if not atomic:
            pipe = self.redis.pipeline(transaction=False)
            for op in ops:
                self._queue_write(pipe, op, check=True)
            return [
                self._result(op, r) for op, r in zip(ops, pipe.execute(), strict=True)
            ]
        _check_atomic(ops)
        conditional = [op for op in ops if _is_conditional(op)]
        watched = sorted({k for op in conditional for k in self._data_keys(op.key)})
        with self.redis.pipeline(transaction=True) as pipe:
            while True:
                try:
                    if watched:
                        # Check every condition under WATCH, then apply the batch
                        # unconditionally in MULTI. EXEC fails if a watched key changed.
                        pipe.watch(*watched)
                        currents = self._currents(pipe, conditional)
                        if any(
                            op.expect != cur
                            for op, cur in zip(conditional, currents, strict=True)
                        ):
                            pipe.unwatch()
                            return _not_applied(ops)
                        pipe.multi()
                    for op in ops:
                        self._queue_write(pipe, op, check=False)
                    replies = pipe.execute()
                    return [
                        self._result(op, r) for op, r in zip(ops, replies, strict=True)
                    ]
                except WatchError:
                    pipe.reset()

    def _currents(self, pipe: Any, ops: list[MapPut | MapDelete]) -> list[bytes | None]:
        currents: list[bytes | None] = []
        for op in ops:
            data, exp = self._data_keys(op.key)
            if is_legacy(op.key):
                currents.append(b"" if pipe.sismember(data, op.field[0]) else None)
            else:
                currents.append(
                    self._scripts["map_get"](
                        keys=[data, exp], args=[encode(op.field)], client=pipe
                    )
                )
        return currents

    def current_time(self) -> float:
        seconds, micros = self.redis.time()
        return seconds + micros / 1e6

    # Journal. Reading a partition costs a round trip, so it is read only when it
    # may have changed: when a wake-up arrives on its channel, when the caller is
    # behind a position that this object has already read, or when its periodic
    # sweep is due.

    def journal_read(
        self,
        cursors: Mapping[KeyPartT, bytes],
        *,
        timeout: float | None = 0,
        limit: int | None = None,
    ) -> tuple[list[Change], dict[KeyPartT, bytes]]:
        # The pubsub connection is not thread-safe.
        with self._journal_lock:
            return super().journal_read(cursors, timeout=timeout, limit=limit)

    def _journal_position(self, partition: KeyPartT) -> bytes:
        last = self.redis.xrevrange(self._journal_key(partition), "+", "-", count=1)
        return _entry_id(last[0][0]) if last else _MIN_ID

    def _journal_fetch(
        self, positions: Mapping[KeyPartT, bytes], limit: int | None
    ) -> _JournalBatch:
        for position in positions.values():
            split_entry_id(position)
        if self._journal_pubsub is None:
            self._journal_pubsub = self.redis.pubsub(ignore_subscribe_messages=True)
        channels = {
            self._wake_channel(p): p
            for p in positions
            if self._wake_channel(p) not in self._wake_partitions
        }
        if channels:
            self._journal_pubsub.subscribe(*channels)
            self._wake_partitions.update(channels)
            self._dirty.update(channels.values())
        while (message := self._journal_pubsub.get_message(timeout=0)) is not None:
            self._note_wakeup(message)
        now = time.monotonic()
        due = [
            p
            for p, position in positions.items()
            if p in self._dirty
            or position < self._read_upto.get(p, _MIN_ID)
            or now - self._last_swept.get(p, 0) > self._sweep_interval
        ]
        changes: list[Change] = []
        cut: set[KeyPartT] = set()
        new_positions = dict(positions)
        if due:
            by_stream = {self._journal_key(p): p for p in due}
            streams = {self._journal_key(p): _stream_id(positions[p]) for p in due}
            for stream, entries in self.redis.xread(streams, count=limit) or []:
                partition = by_stream[stream]
                for stream_id, fields in entries:
                    if limit is not None and len(changes) >= limit:
                        cut.add(partition)
                        break
                    changes.append(
                        _journal_change(
                            int(fields[b"o"]),
                            fields[b"k"],
                            fields.get(b"f"),
                            _entry_id(fields[b"i"]) if b"i" in fields else None,
                            fields.get(b"v"),
                        )
                    )
                    new_positions[partition] = _entry_id(stream_id)
                if limit is not None and len(entries) >= limit:
                    cut.add(partition)
            for p in due:
                self._last_swept[p] = now
                if p not in cut:
                    self._dirty.discard(p)
                    self._read_upto[p] = max(
                        new_positions[p], self._read_upto.get(p, _MIN_ID)
                    )
        return _JournalBatch(changes, new_positions, cut)

    def _journal_wait(self, batch: _JournalBatch, timeout: float | None) -> None:
        wait = self._sweep_interval
        if timeout is not None:
            wait = min(timeout, wait)
        message = self._journal_pubsub.get_message(timeout=max(wait, 0.001))
        if message is not None:
            self._note_wakeup(message)

    def _note_wakeup(self, message: dict) -> None:
        if message["type"] == "message":
            partition = self._wake_partitions.get(message["channel"])
            if partition is not None:
                self._dirty.add(partition)

    def close(self) -> None:
        super().close()
        with self._journal_lock:
            if self._journal_pubsub is not None:
                self._journal_pubsub.close()
                self._journal_pubsub = None
            for state in (
                self._wake_partitions,
                self._dirty,
                self._read_upto,
                self._last_swept,
            ):
                state.clear()

    # The old listener API, fed by the firehose channel.

    def _handle_message(self, message: dict) -> None:
        # other message types include "subscribe" and "unsubscribe". these are
        # sent to the client, but not to the pubsub channel.
        assert message["type"] == "message"
        data = message["data"]
        if data[:1] in (b"A", b"D"):  # a log entry, appended or deleted
            size, rest = data[1:].split(b":", 1)
            key, rest = rest[: int(size)], rest[int(size) :]
            size, rest = rest.split(b":", 1)
            stream_id, value = rest[: int(size)], rest[int(size) :]
            kind = "save" if data[:1] == b"A" else "delete"
            self._broadcast_change((kind, (key, _entry_id(stream_id) + value)))
            return
        event_type, (key, value) = json.loads(data)
        self._broadcast_change(
            (
                event_type,
                (self._decode(key), None if value is None else self._decode(value)),
            )
        )

    def _start_listening(self) -> None:
        self._pubsub = self.redis.pubsub(ignore_subscribe_messages=True)
        self._pubsub.subscribe(**{self.listener_channel: self._handle_message})
        # redis-py only calls handlers while something reads messages.
        self._listen_thread = self._pubsub.run_in_thread(sleep_time=0.01, daemon=True)

    def _stop_listening(self) -> None:
        self._listen_thread.stop()
        self._listen_thread.join()
        self._pubsub.close()
        self._pubsub = self._listen_thread = None
