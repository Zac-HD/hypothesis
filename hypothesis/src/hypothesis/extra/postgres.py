# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""A database backend for Postgres 13 or later, using :pypi:`psycopg` 3.2 or later.

See section 5 of ``guides/database-design.md``. In short:

* Every write to a partition happens in a transaction that holds an advisory
  lock for that partition, taken before the transaction's first write. Writes
  to a partition are therefore ordered by transaction id, and conditions can be
  checked with plain reads.
* Journal rows are read in ``(xid, id)`` order, and only once every transaction
  with a smaller id has finished. Rows are never skipped, and rows of one
  partition arrive in the order they were written.
* ``pg_notify`` wakes readers, on one of 64 channels per namespace.
"""

import os
import struct
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

import psycopg

from hypothesis.database import (
    INLINE_VALUE_LIMIT,
    Change,
    LogAppend,
    LogTrim,
    MapClear,
    MapDelete,
    MapGet,
    MapItems,
    MapPut,
    ReadOpT,
    WriteOpT,
    _check_atomic,
    _Connections,
    _is_conditional,
    _journal_change,
    _JournalBatch,
    _ListenerThread,
    _NativeDatabase,
    _not_applied,
    _sql_batch,
    _unpack_position,
)
from hypothesis.internal.dbcodec import (
    KeyPartT,
    decode,
    encode,
    partition_hash,
    short_hash,
)

_MAX_ID = 2**63 - 1
_CHANNELS = 64
_NOW = "extract(epoch from now())"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS hypothesis_maps (
    ns text NOT NULL, kh bytea NOT NULL, fh bytea NOT NULL, key bytea NOT NULL,
    field bytea NOT NULL, value bytea NOT NULL, exp double precision,
    PRIMARY KEY (ns, kh, fh)
);
CREATE INDEX IF NOT EXISTS hypothesis_maps_exp ON hypothesis_maps (exp) WHERE exp IS NOT NULL;
CREATE TABLE IF NOT EXISTS hypothesis_logs (
    ns text NOT NULL, kh bytea NOT NULL, id bytea NOT NULL, value bytea NOT NULL,
    PRIMARY KEY (ns, kh, id)
);
CREATE TABLE IF NOT EXISTS hypothesis_log_meta (
    ns text NOT NULL, kh bytea NOT NULL, last_ms bigint NOT NULL, last_seq bigint NOT NULL,
    count bigint NOT NULL, ttl double precision, PRIMARY KEY (ns, kh)
);
CREATE TABLE IF NOT EXISTS hypothesis_journal (
    xid xid8 NOT NULL DEFAULT pg_current_xact_id(), id bigserial NOT NULL,
    ns text NOT NULL, at double precision NOT NULL, ph bytea NOT NULL,
    op smallint NOT NULL, key bytea NOT NULL, field bytea, eid bytea, value bytea,
    PRIMARY KEY (xid, id)
);
CREATE INDEX IF NOT EXISTS hypothesis_journal_at ON hypothesis_journal (at);
"""

_LIVE = f"(exp IS NULL OR exp > {_NOW})"

# Applies every operation for one partition, in one statement, which is one
# transaction. Returns NULL if the batch is atomic and a condition failed.
_APPLY_FUNCTION = """
CREATE OR REPLACE FUNCTION hypothesis_apply_v2(
    p_ns text, p_ph bytea, p_lock bigint, p_channel text, p_sync boolean,
    p_atomic boolean, p_inline int, p_ops smallint[], p_keys bytea[], p_khs bytea[],
    p_fields bytea[], p_fhs bytea[], p_values bytea[], p_ttls float8[],
    p_counts int[], p_befores bytea[], p_ids bytea[], p_modes smallint[],
    p_expects bytea[]
) RETURNS bytea[] LANGUAGE plpgsql AS $$
DECLARE
    results bytea[] := '{}';
    now_s float8 := extract(epoch from now());
    n int := coalesce(array_length(p_ops, 1), 0);
    cur bytea;
    failed boolean;
BEGIN
    IF NOT p_sync THEN
        PERFORM set_config('synchronous_commit', 'off', true);
    END IF;
    PERFORM pg_advisory_xact_lock(p_lock);
    -- Check every condition first, so that an atomic batch can apply nothing.
    FOR i IN 1..n LOOP
        failed := false;
        IF p_modes[i] > 0 THEN
            SELECT value INTO cur FROM hypothesis_maps
            WHERE ns = p_ns AND kh = p_khs[i] AND fh = p_fhs[i] AND key = p_keys[i]
            AND field = p_fields[i] AND (exp IS NULL OR exp > now_s);
            failed := (p_modes[i] = 1 AND FOUND)
                OR (p_modes[i] = 2 AND (NOT FOUND OR cur IS DISTINCT FROM p_expects[i]));
        END IF;
        IF failed AND p_atomic THEN
            RETURN NULL;
        ELSIF failed THEN
            results := array_append(results, '\\x00'::bytea);
        ELSIF NOT p_atomic THEN
            results := array_append(results, hypothesis_apply_one_v2(
                p_ns, p_ph, now_s, p_inline, p_ops[i], p_keys[i], p_khs[i], p_fields[i],
                p_fhs[i], p_values[i], p_ttls[i], p_counts[i], p_befores[i], p_ids[i]));
        END IF;
    END LOOP;
    IF p_atomic THEN
        FOR i IN 1..n LOOP
            results := array_append(results, hypothesis_apply_one_v2(
                p_ns, p_ph, now_s, p_inline, p_ops[i], p_keys[i], p_khs[i], p_fields[i],
                p_fhs[i], p_values[i], p_ttls[i], p_counts[i], p_befores[i], p_ids[i]));
        END LOOP;
    END IF;
    -- Wake readers only if this transaction wrote to the journal.
    IF EXISTS (
        SELECT 1 FROM hypothesis_journal WHERE xid = pg_current_xact_id_if_assigned()
    ) THEN
        PERFORM pg_notify(p_channel, '');
    END IF;
    RETURN results;
END $$;

-- p_ids holds the ids of entries to delete, 16 bytes each, one after another.
CREATE OR REPLACE FUNCTION hypothesis_apply_one_v2(
    p_ns text, p_ph bytea, now_s float8, p_inline int, p_op smallint, p_key bytea,
    p_kh bytea, p_field bytea, p_fh bytea, p_value bytea, p_ttl float8,
    p_count int, p_before bytea, p_ids bytea
) RETURNS bytea LANGUAGE plpgsql AS $$
DECLARE
    cur bytea;
    present boolean;
    meta record;
    new_id bytea;
    cut bytea;
    doomed bytea;
    removed bigint := 0;
    k bigint;
BEGIN
    IF p_op IN (1, 2) THEN
        SELECT value INTO cur FROM hypothesis_maps
        WHERE ns = p_ns AND kh = p_kh AND fh = p_fh AND key = p_key AND field = p_field
        AND (exp IS NULL OR exp > now_s);
        present := FOUND;
    END IF;
    IF p_op = 1 THEN
        INSERT INTO hypothesis_maps (ns, kh, fh, key, field, value, exp)
        VALUES (p_ns, p_kh, p_fh, p_key, p_field, p_value, now_s + p_ttl)
        ON CONFLICT (ns, kh, fh) DO UPDATE SET value = EXCLUDED.value, exp = EXCLUDED.exp
        WHERE hypothesis_maps.value IS DISTINCT FROM EXCLUDED.value
        OR hypothesis_maps.exp IS DISTINCT FROM EXCLUDED.exp;
        IF NOT present OR cur IS DISTINCT FROM p_value THEN
            INSERT INTO hypothesis_journal (ns, at, ph, op, key, field, value)
            VALUES (p_ns, now_s, p_ph, 1, p_key, p_field,
                    CASE WHEN length(p_value) <= p_inline THEN p_value END);
        END IF;
        RETURN '\\x01'::bytea;
    ELSIF p_op = 2 THEN
        DELETE FROM hypothesis_maps
        WHERE ns = p_ns AND kh = p_kh AND fh = p_fh AND key = p_key AND field = p_field;
        IF NOT present THEN
            RETURN '\\x00'::bytea;
        END IF;
        INSERT INTO hypothesis_journal (ns, at, ph, op, key, field)
        VALUES (p_ns, now_s, p_ph, 2, p_key, p_field);
        RETURN '\\x01'::bytea;
    ELSIF p_op = 3 THEN
        DELETE FROM hypothesis_maps WHERE ns = p_ns AND kh = p_kh AND key = p_key;
        IF FOUND THEN
            INSERT INTO hypothesis_journal (ns, at, ph, op, key)
            VALUES (p_ns, now_s, p_ph, 3, p_key);
        END IF;
        RETURN NULL;
    ELSIF p_op = 4 THEN
        INSERT INTO hypothesis_log_meta AS m (ns, kh, last_ms, last_seq, count, ttl)
        VALUES (p_ns, p_kh, (extract(epoch from clock_timestamp()) * 1000)::bigint, 0, 1, p_ttl)
        ON CONFLICT (ns, kh) DO UPDATE SET
            last_seq = CASE WHEN EXCLUDED.last_ms > m.last_ms THEN 0 ELSE m.last_seq + 1 END,
            last_ms = GREATEST(EXCLUDED.last_ms, m.last_ms),
            count = m.count + 1,
            ttl = EXCLUDED.ttl
        RETURNING m.last_ms, m.last_seq, m.count INTO meta;
        new_id := int8send(meta.last_ms) || int8send(meta.last_seq);
        INSERT INTO hypothesis_logs (ns, kh, id, value) VALUES (p_ns, p_kh, new_id, p_value);
        INSERT INTO hypothesis_journal (ns, at, ph, op, key, eid, value)
        VALUES (p_ns, now_s, p_ph, 4, p_key, new_id,
                CASE WHEN length(p_value) <= p_inline THEN p_value END);
        -- Trim in bulk, once the log is a quarter over its maximum length.
        IF p_count IS NOT NULL AND meta.count > p_count + greatest(1, p_count / 4) THEN
            SELECT id INTO cut FROM hypothesis_logs WHERE ns = p_ns AND kh = p_kh
            ORDER BY id DESC OFFSET p_count LIMIT 1;
            IF FOUND THEN
                DELETE FROM hypothesis_logs WHERE ns = p_ns AND kh = p_kh AND id <= cut;
                GET DIAGNOSTICS k = ROW_COUNT;
                UPDATE hypothesis_log_meta SET count = count - k WHERE ns = p_ns AND kh = p_kh;
            END IF;
        END IF;
        RETURN new_id;
    END IF;
    -- Entries deleted by id are journaled. Trims are not.
    FOR j IN 0 .. coalesce(length(p_ids), 0) / 16 - 1 LOOP
        doomed := substring(p_ids FROM j * 16 + 1 FOR 16);
        DELETE FROM hypothesis_logs WHERE ns = p_ns AND kh = p_kh AND id = doomed
        RETURNING value INTO cur;
        IF FOUND THEN
            removed := removed + 1;
            INSERT INTO hypothesis_journal (ns, at, ph, op, key, eid, value)
            VALUES (p_ns, now_s, p_ph, 2, p_key, doomed,
                    CASE WHEN length(cur) <= p_inline THEN cur END);
        END IF;
    END LOOP;
    IF p_before IS NOT NULL THEN
        DELETE FROM hypothesis_logs WHERE ns = p_ns AND kh = p_kh AND id < p_before;
        GET DIAGNOSTICS k = ROW_COUNT;
        removed := removed + k;
    END IF;
    IF p_count IS NOT NULL THEN
        SELECT id INTO cut FROM hypothesis_logs WHERE ns = p_ns AND kh = p_kh
        ORDER BY id DESC OFFSET p_count LIMIT 1;
        IF FOUND THEN
            DELETE FROM hypothesis_logs WHERE ns = p_ns AND kh = p_kh AND id <= cut;
            GET DIAGNOSTICS k = ROW_COUNT;
            removed := removed + k;
        END IF;
    END IF;
    UPDATE hypothesis_log_meta SET count = count - removed WHERE ns = p_ns AND kh = p_kh;
    RETURN convert_to(removed::text, 'UTF8');
END $$;
"""

_APPLY = """
SELECT hypothesis_apply_v2(
    %s, %s, %s, %s, %s, %s, %s, %s::smallint[], %s::bytea[], %s::bytea[],
    %s::bytea[], %s::bytea[], %s::bytea[], %s::float8[], %s::int[], %s::bytea[],
    %s::bytea[], %s::smallint[], %s::bytea[]
)
"""

_OP_CODES = {MapPut: 1, MapDelete: 2, MapClear: 3, LogAppend: 4, LogTrim: 5}

_JOURNAL_READ = """
WITH s AS (SELECT pg_snapshot_xmin(pg_current_snapshot()) AS xmin)
SELECT s.xmin::text::bigint, j.xid::text::bigint, j.id, j.ph, j.op, j.key, j.field, j.eid, j.value
FROM s LEFT JOIN LATERAL (
    SELECT * FROM hypothesis_journal
    WHERE (xid, id) > (%(xid)s::text::xid8, %(id)s) AND xid < s.xmin
    AND ns = %(ns)s {partitions}
    ORDER BY xid, id LIMIT %(limit)s
) j ON true
"""


class _PostgresListener(_ListenerThread):
    """Follows every partition in the namespace, for the old listener API."""

    db: "PostgresExampleDatabase"

    def __init__(self, db: "PostgresExampleDatabase") -> None:
        super().__init__(db)
        self.conn = db._connect()
        self.listen_conn = psycopg.connect(db.conninfo, autocommit=True)
        for i in range(_CHANNELS):
            self.listen_conn.execute(f"LISTEN {db._channel_name(i)}")
        self.position = db._xmin_position(self.conn)

    def fetch(self) -> list[Change]:
        _, rows = self.db._read_journal(self.conn, self.position, None, 1000)
        if rows:
            self.position = (rows[-1][0], rows[-1][1])
        return [_row_change(row) for row in rows]

    def wait(self) -> None:
        for _ in self.listen_conn.notifies(timeout=0.25, stop_after=1):
            pass

    def release(self) -> None:
        self.conn.close()
        self.listen_conn.close()


class PostgresExampleDatabase(_NativeDatabase):
    """Store examples in a Postgres database, given a libpq connection string.

    Every ``namespace`` is separate. The tables are created if they do not exist.
    Each thread that uses the database opens its own connection, and following
    the journal opens one more.
    """

    _listener_thread_class = _PostgresListener

    def __init__(
        self,
        conninfo: str,
        *,
        namespace: str = "default",
        journal_retention: float = 300.0,
        synchronous_commit: bool = False,
    ) -> None:
        super().__init__()
        self.conninfo = conninfo
        self.namespace = namespace
        # Without synchronous commit, a crash can lose the last fraction of a
        # second of writes, but never corrupts anything. That suits a cache.
        self.synchronous_commit = synchronous_commit
        self.journal_retention = journal_retention
        self._ns_hash = short_hash(namespace.encode())[:4].hex()
        self._connections = _Connections(self._connect)
        self._next_cleanup = time.time() + 1
        self._journal_lock = threading.Lock()
        self._journal_conn: Any = None
        self._journal_pid = 0
        self._channels: set[str] = set()

    def __repr__(self) -> str:
        return (
            f"PostgresExampleDatabase({self.conninfo!r}, namespace={self.namespace!r})"
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, PostgresExampleDatabase)
            and self.conninfo == other.conninfo
            and self.namespace == other.namespace
        )

    def __getstate__(self) -> dict[str, Any]:
        return {
            "conninfo": self.conninfo,
            "namespace": self.namespace,
            "journal_retention": self.journal_retention,
            "synchronous_commit": self.synchronous_commit,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(**state)  # type: ignore

    @property
    def capabilities(self) -> frozenset[str]:
        return frozenset(
            {"native", "atomic", "journal", "blocking", "shared", "ttl", "server_time"}
        )

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self.conninfo, autocommit=True)
        if self._schema_missing(conn):
            with conn.transaction():
                # Only one process creates the schema. Skip the statements when
                # they would do nothing, because CREATE INDEX IF NOT EXISTS still
                # locks its table, and can deadlock with processes that are writing.
                conn.execute("SELECT pg_advisory_xact_lock(7469821)")
                if self._schema_missing(conn):
                    conn.execute(_SCHEMA)
                    conn.execute(_APPLY_FUNCTION)
        return conn

    @staticmethod
    def _schema_missing(conn: psycopg.Connection) -> bool:
        # Reads the catalog, which takes no locks on the tables.
        row = conn.execute(
            "SELECT to_regclass('hypothesis_journal_at') IS NULL "
            "OR to_regproc('hypothesis_apply_one_v2') IS NULL"
        ).fetchone()
        return bool(row[0])

    def _conn(self) -> psycopg.Connection:
        return self._connections.get()

    def _channel_name(self, index: int) -> str:
        return f"hypothesis_{self._ns_hash}_{index}"

    def _channel(self, ph: bytes) -> str:
        return self._channel_name(ph[0] % _CHANNELS)

    def _lock_id(self, ph: bytes) -> int:
        return int.from_bytes(
            short_hash(self.namespace.encode() + ph)[:8], "big", signed=True
        )

    # Reads

    def read_many(self, ops: Sequence[ReadOpT]) -> list[Any]:
        ops = list(ops)
        conn = self._conn()
        cursors = []
        with conn.pipeline():
            for op in ops:
                ek = encode(op.key)
                params: dict[str, Any] = {
                    "ns": self.namespace,
                    "key": ek,
                    "kh": short_hash(ek),
                }
                if isinstance(op, MapGet):
                    ef = encode(op.field)
                    params.update(field=ef, fh=short_hash(ef))
                    sql = (
                        "SELECT value FROM hypothesis_maps WHERE ns = %(ns)s AND kh = %(kh)s "
                        f"AND fh = %(fh)s AND key = %(key)s AND field = %(field)s AND {_LIVE}"
                    )
                elif isinstance(op, MapItems):
                    sql = (
                        "SELECT field, value FROM hypothesis_maps WHERE ns = %(ns)s "
                        f"AND kh = %(kh)s AND key = %(key)s AND {_LIVE}"
                    )
                else:
                    sql = "SELECT id, value FROM hypothesis_logs WHERE ns = %(ns)s AND kh = %(kh)s"
                    if op.after is not None:
                        sql += " AND id > %(after)s"
                    if op.before is not None:
                        sql += " AND id < %(before)s"
                    sql += " ORDER BY id DESC" if op.reverse else " ORDER BY id"
                    sql += " LIMIT %(limit)s"
                    params.update(after=op.after, before=op.before, limit=op.limit)
                cursors.append(conn.execute(sql, params))
        results: list[Any] = []
        for op, cursor in zip(ops, cursors, strict=True):
            rows = cursor.fetchall()
            if isinstance(op, MapGet):
                results.append(bytes(rows[0][0]) if rows else None)
            elif isinstance(op, MapItems):
                prefix = encode(op.prefix)
                results.append(
                    {
                        decode(bytes(field)): bytes(value)
                        for field, value in rows
                        if bytes(field).startswith(prefix)
                    }
                )
            else:
                results.append([(bytes(i), bytes(v)) for i, v in rows])
        return results

    # Writes

    def _apply_args(self, ph: bytes, ops: list[WriteOpT], atomic: bool) -> list[Any]:
        """The arguments to hypothesis_apply_v2, with one array per column."""
        rows = []
        for op in ops:
            ek = encode(op.key)
            ef = encode(op.field) if isinstance(op, (MapPut, MapDelete)) else None
            mode, expect = 0, None
            if _is_conditional(op):
                mode, expect = (1, None) if op.expect is None else (2, op.expect)
            ids = op.ids if isinstance(op, LogTrim) else ()
            rows.append(
                (
                    _OP_CODES[type(op)],
                    ek,
                    short_hash(ek),
                    ef,
                    None if ef is None else short_hash(ef),
                    getattr(op, "value", None),
                    getattr(op, "ttl", None),
                    getattr(op, "maxlen", None),
                    getattr(op, "before", None),
                    b"".join(ids) if ids else None,
                    mode,
                    expect,
                )
            )
        return [
            self.namespace,
            ph,
            self._lock_id(ph),
            self._channel(ph),
            self.synchronous_commit,
            atomic,
            INLINE_VALUE_LIMIT,
            *map(list, zip(*rows, strict=True)),
        ]

    def write_many(self, ops: Sequence[WriteOpT], *, atomic: bool = False) -> list[Any]:
        ops = list(ops)
        if not ops:
            return []
        if atomic:
            _check_atomic(ops)
        groups: dict[bytes, list[int]] = {}
        for i, op in enumerate(ops):
            groups.setdefault(partition_hash(op.key[0]), []).append(i)
        conn = self._conn()
        cursors = {}
        with conn.pipeline():
            # One statement, and so one transaction, per partition, so that no
            # partition's lock is held for long.
            for ph, indices in groups.items():
                args = self._apply_args(ph, [ops[i] for i in indices], atomic)
                cursors[ph] = conn.execute(_APPLY, args)
        results: list[Any] = [None] * len(ops)
        for ph, indices in groups.items():
            replies = cursors[ph].fetchone()[0]
            if replies is None:
                return _not_applied(ops)
            for i, reply in zip(indices, replies, strict=True):
                results[i] = self._result(ops[i], reply)
        if time.time() > self._next_cleanup:
            self._cleanup(conn)
        return results

    @staticmethod
    def _result(op: WriteOpT, reply: Any) -> Any:
        if isinstance(op, (MapPut, MapDelete)):
            return bytes(reply) == b"\x01"
        if isinstance(op, LogAppend):
            return bytes(reply)
        if isinstance(op, LogTrim):
            return int(bytes(reply))
        return None

    def _cleanup(self, conn: psycopg.Connection) -> None:
        self._next_cleanup = time.time() + min(10.0, self.journal_retention / 4)
        params = {"ns": self.namespace, "retention": self.journal_retention}
        with conn.transaction():
            conn.execute(
                f"DELETE FROM hypothesis_journal WHERE at < {_NOW} - %(retention)s",
                params,
            )
            conn.execute(
                f"DELETE FROM hypothesis_maps WHERE ns = %(ns)s AND exp < {_NOW}",
                params,
            )
            conn.execute(
                f"""
                WITH d AS (
                    DELETE FROM hypothesis_logs l USING hypothesis_log_meta m
                    WHERE l.ns = %(ns)s AND m.ns = l.ns AND m.kh = l.kh AND m.ttl IS NOT NULL
                    AND l.id < int8send((({_NOW} - m.ttl) * 1000)::bigint) || int8send(0::bigint)
                    RETURNING l.kh
                )
                UPDATE hypothesis_log_meta m SET count = m.count - c.n
                FROM (SELECT kh, count(*) AS n FROM d GROUP BY kh) c
                WHERE m.ns = %(ns)s AND m.kh = c.kh
                """,
                params,
            )

    def current_time(self) -> float:
        row = (
            self._conn()
            .execute("SELECT extract(epoch from clock_timestamp())")
            .fetchone()
        )
        return float(row[0])

    # Journal

    def _xmin_position(self, conn: psycopg.Connection) -> tuple[int, int]:
        row = conn.execute(
            "SELECT pg_snapshot_xmin(pg_current_snapshot())::text::bigint"
        ).fetchone()
        # Every transaction before xmin has finished, so it is safe to skip.
        return (row[0] - 1, _MAX_ID)

    def _read_journal(
        self,
        conn: psycopg.Connection,
        low: tuple[int, int],
        hashes: list[bytes] | None,
        limit: int | None,
    ) -> tuple[int, list[tuple[Any, ...]]]:
        partitions = "" if hashes is None else "AND ph = ANY(%(phs)s)"
        rows = conn.execute(
            _JOURNAL_READ.format(partitions=partitions),
            {
                "xid": low[0],
                "id": low[1],
                "ns": self.namespace,
                "phs": hashes,
                "limit": limit,
            },
        ).fetchall()
        xmin = rows[0][0]
        return xmin, [row[1:] for row in rows if row[2] is not None]

    def _listen(self, channels: set[str]) -> None:
        if self._journal_conn is None or self._journal_pid != os.getpid():
            self._journal_conn = psycopg.connect(self.conninfo, autocommit=True)
            self._journal_pid = os.getpid()
            self._channels = set()
        for channel in sorted(channels - self._channels):
            self._journal_conn.execute(f"LISTEN {channel}")
            self._channels.add(channel)

    def journal_read(
        self,
        cursors: Mapping[KeyPartT, bytes],
        *,
        timeout: float | None = 0,
        limit: int | None = None,
    ) -> tuple[list[Change], dict[KeyPartT, bytes]]:
        # One connection listens for wake-ups, and it is not thread-safe.
        with self._journal_lock:
            return super().journal_read(cursors, timeout=timeout, limit=limit)

    def _journal_position(self, partition: KeyPartT) -> bytes:
        return struct.pack(">qq", *self._xmin_position(self._conn()))

    def _journal_fetch(
        self, positions: Mapping[KeyPartT, bytes], limit: int | None
    ) -> _JournalBatch:
        unpacked = {p: _unpack_position(">qq", pos) for p, pos in positions.items()}
        by_hash = {partition_hash(p): p for p in unpacked}
        # LISTEN before reading, so that no wake-up is missed.
        self._listen({self._channel(ph) for ph in by_hash})
        xmin, rows = self._read_journal(
            self._conn(), min(unpacked.values()), list(by_hash), limit
        )
        entries = [
            ((row[0], row[1]), by_hash[bytes(row[2])], _row_change(row)) for row in rows
        ]
        return _sql_batch(">qq", unpacked, entries, (xmin - 1, _MAX_ID), limit)

    def _journal_wait(self, batch: _JournalBatch, timeout: float | None) -> None:
        wait = 1.0 if timeout is None else min(timeout, 1.0)
        if any(True for _ in self._journal_conn.notifies(timeout=wait, stop_after=1)):
            # Take every other pending wake-up too, so that one read serves them all.
            while any(
                True for _ in self._journal_conn.notifies(timeout=0, stop_after=1)
            ):
                pass

    def close(self) -> None:
        super().close()
        self._connections.close()
        with self._journal_lock:
            if self._journal_conn is not None and self._journal_pid == os.getpid():
                self._journal_conn.close()
            self._journal_conn = None
            self._channels = set()


def _row_change(row: tuple[Any, ...]) -> Change:
    _xid, _id, _ph, op, key, field, eid, value = row
    return _journal_change(op, bytes(key), _bytes(field), _bytes(eid), _bytes(value))


def _bytes(value: Any) -> bytes | None:
    return None if value is None else bytes(value)
