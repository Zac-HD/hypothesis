# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Serves one database to the other processes on a machine.

See section 6 of ``guides/database-design.md``. Two threads do the work:

* The loop thread reads every connection. Each time round, it applies the writes
  it has received as one ``write_many``, then answers the reads as one
  ``read_many``. A client's writes are therefore applied before its later reads.
* The journal thread follows the backend's journal for every partition that a
  client follows, and keeps the changes in memory.

The first version had a thread per connection, and then a separate writer thread.
The benchmarks showed that passing work between threads cost more than the work.
"""

import os
import queue
import selectors
import socket
import threading
import time
import traceback
from collections.abc import Callable
from multiprocessing.connection import Client, Connection, Listener, Pipe, wait
from typing import Any

from hypothesis.database import (
    ExampleDatabase,
    JournalCursorExpired,
    RemoteDatabase,
    WriteOpT,
    _JournalBuffer,
)
from hypothesis.internal.compat import WINDOWS
from hypothesis.internal.dbcodec import KeyPartT


class _JournalWait:
    def __init__(
        self, cursors: dict, limit: int | None, deadline: float | None
    ) -> None:
        self.cursors = cursors
        self.limit = limit
        self.deadline = deadline
        self.changes: list = []
        self.ready_at: float | None = None
        # The buffer's sequence number at the last read, and when to read anyway.
        self.seen = -1
        self.next_read = 0.0


class _Client:
    """One connection. The client waits for each reply, so at most one request
    that needs a reply is outstanding."""

    def __init__(self, conn: Connection) -> None:
        self.conn = conn
        self.errors: list[BaseException] = []
        self.pending: tuple[str, tuple] | None = None
        self.journal: _JournalWait | None = None


class _JournalHub:
    """Follows the backend's journal for every partition that a client follows."""

    def __init__(
        self,
        db: ExampleDatabase,
        poll_timeout: float,
        linger: float,
        wake: Callable[[], None],
    ) -> None:
        self.db = db
        self.poll_timeout = poll_timeout
        self.linger = linger
        self.wake_loop = wake
        self.buffer = _JournalBuffer(db.journal_retention)
        self.backend_cursors: dict[KeyPartT, bytes] = {}
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.closed = threading.Event()

    def head(self, partition: KeyPartT) -> bytes:
        with self.lock:
            if partition not in self.backend_cursors:
                # Take the backend's head before registering the partition, so the
                # hub cannot deliver a change that the backend head also covers.
                self.backend_cursors[partition] = self.db.journal_head(partition)
                self.wake.set()
            return self.buffer.head(partition)

    def run(self) -> None:
        next_prune = time.monotonic() + 10
        while not self.closed.is_set():
            with self.lock:
                cursors = dict(self.backend_cursors)
            if not cursors:
                self.wake.wait(1.0)
                self.wake.clear()
                continue
            try:
                changes, new = self.db.journal_read(
                    cursors, timeout=self.poll_timeout, limit=1000
                )
            except JournalCursorExpired as err:
                with self.lock:
                    for partition in err.partitions:
                        self.backend_cursors[partition] = self.db.journal_head(
                            partition
                        )
                self.buffer.expire_partitions(err.partitions)
                self.wake_loop()
                continue
            except Exception:
                if self.closed.is_set():
                    return
                traceback.print_exc()
                time.sleep(1)
                continue
            if changes:
                self.buffer.add(changes)
                self.wake_loop()
                # Let more changes arrive, then read them together.
                time.sleep(self.linger)
            with self.lock:
                for partition, cursor in new.items():
                    if partition in self.backend_cursors:
                        self.backend_cursors[partition] = cursor
                if time.monotonic() > next_prune:
                    # The buffer forgets partitions that nobody has read for a while.
                    next_prune = time.monotonic() + 10
                    for partition in list(self.backend_cursors):
                        if partition not in self.buffer.queues:
                            del self.backend_cursors[partition]


class DatabaseServer:
    """Serves ``db`` on a local socket, from background threads.

    ``client()`` returns a :class:`~hypothesis.database.RemoteDatabase`, which
    can be pickled and passed to subprocesses.
    """

    def __init__(
        self,
        db: ExampleDatabase,
        *,
        address: Any = None,
        authkey: bytes | None = None,
        max_batch: int = 500,
        poll_timeout: float = 0.5,
        linger: float = 0.005,
    ) -> None:
        self.db = db
        self.authkey = authkey or os.urandom(32)
        self._listener = Listener(
            address, family="AF_PIPE" if WINDOWS else "AF_UNIX", authkey=self.authkey
        )
        self.address = self._listener.address
        self._max_batch = max_batch
        self._linger = linger
        self._writes: list[tuple[_Client, list[WriteOpT]]] = []
        self._new_conns: queue.Queue[Connection] = queue.Queue()
        # A pipe, not a socket, because on Windows wait() reads each object with an
        # overlapped ReadFile, which a socket handle does not support.
        self._wake_recv, self._wake_send = Pipe(duplex=False)
        self._wake_lock = threading.Lock()
        # One selector, so that each pass does not register every connection
        # again. A selector cannot wait for a named pipe, so Windows uses wait().
        self._selector = None if WINDOWS else selectors.DefaultSelector()
        if self._selector is not None:
            self._selector.register(self._wake_recv, selectors.EVENT_READ)
        self._closed = threading.Event()
        self._hub = _JournalHub(db, poll_timeout, linger, self._wake)
        self._threads = [
            threading.Thread(target=target, daemon=True, name=name)
            for target, name in [
                (self._accept_loop, "hypothesis-db-accept"),
                (self._loop, "hypothesis-db-loop"),
                (self._hub.run, "hypothesis-db-journal"),
            ]
        ]
        for thread in self._threads:
            thread.start()

    def __repr__(self) -> str:
        return f"DatabaseServer({self.db!r}, address={self.address!r})"

    def client(self) -> RemoteDatabase:
        return RemoteDatabase(self.address, self.authkey)

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._hub.closed.set()
        self._wake()
        self._interrupt_accept()
        self._listener.close()
        # The journal thread may be waiting for the backend, so it is not joined.
        # It stops within poll_timeout.
        for thread in self._threads[:2]:
            thread.join(timeout=10)
        self._wake_recv.close()
        self._wake_send.close()

    def __enter__(self) -> "DatabaseServer":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _wake(self) -> None:
        # Two threads wake the loop, and a Connection frames each message, so the
        # send is locked to keep two frames from interleaving.
        try:
            with self._wake_lock:
                self._wake_send.send_bytes(b"")
        except OSError:
            pass  # the server is closed

    def _interrupt_accept(self) -> None:
        # Closing a listening socket does not interrupt accept(), so connect to it.
        # A raw connection is enough, and unlike Client() it never waits for a
        # reply, which would hang if the accept thread had already stopped.
        try:
            if WINDOWS:  # pragma: no cover
                Client(self.address, authkey=self.authkey).close()
            else:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                    sock.connect(self.address)
        except OSError:  # pragma: no cover
            pass

    def _accept_loop(self) -> None:
        while not self._closed.is_set():
            try:
                conn = self._listener.accept()
            except (OSError, EOFError):
                continue
            if self._closed.is_set():
                conn.close()
                return
            self._new_conns.put(conn)
            self._wake()

    def _apply_writes(self) -> None:
        """Apply the queued writes, in batches of at most max_batch operations."""
        while self._writes:
            batch, count = [], 0
            while self._writes and count < self._max_batch:
                batch.append(self._writes.pop(0))
                count += len(batch[-1][1])
            try:
                self.db.write_many([op for _, ops in batch for op in ops])
            except Exception:
                # Apply each client's writes separately, to find whose failed.
                for client, ops in batch:
                    try:
                        self.db.write_many(ops)
                    except Exception as err:
                        client.errors.append(err)

    # The loop thread

    def _loop(self) -> None:
        clients: dict[Connection, _Client] = {}
        try:
            while not self._closed.is_set():
                while not self._new_conns.empty():
                    conn = self._new_conns.get()
                    clients[conn] = _Client(conn)
                    if self._selector is not None:
                        self._selector.register(conn, selectors.EVENT_READ)
                for ready in self._ready(clients):
                    if ready is self._wake_recv:
                        try:
                            while self._wake_recv.poll(0):
                                self._wake_recv.recv_bytes()
                        except OSError:
                            pass
                    elif clients[ready].pending is None:
                        self._receive(clients, clients[ready])
                    else:
                        # A client waits for each reply before it sends more, so
                        # a connection with a request pending is readable only
                        # when the client has gone.
                        self._drop(clients, clients[ready])
                # While this runs, clients wait, and their writes queue up in the
                # sockets. That is the backpressure.
                self._apply_writes()
                self._answer(clients)
        finally:
            for conn in clients:
                conn.close()
            if self._selector is not None:
                self._selector.close()

    def _ready(self, clients: dict[Connection, _Client]) -> list[Any]:
        timeout = self._timeout(clients)
        if self._selector is None:  # pragma: no cover
            idle = [c.conn for c in clients.values() if c.pending is None]
            return wait([*idle, self._wake_recv], timeout)
        return [key.fileobj for key, _ in self._selector.select(timeout)]

    def _drop(self, clients: dict[Connection, _Client], client: _Client) -> None:
        del clients[client.conn]
        if self._selector is not None:
            self._selector.unregister(client.conn)
        client.conn.close()

    @staticmethod
    def _timeout(clients: dict[Connection, _Client]) -> float:
        now = time.monotonic()
        timeout = 1.0
        for client in clients.values():
            waiting = client.journal
            if waiting is not None:
                until = (
                    waiting.ready_at
                    if waiting.ready_at is not None
                    else waiting.deadline
                )
                if until is not None:
                    timeout = min(timeout, until - now)
        return max(0.0, timeout)

    def _receive(self, clients: dict[Connection, _Client], client: _Client) -> None:
        # Take every message that has arrived, up to the first that needs a reply.
        while client.pending is None:
            try:
                method, args, writes = client.conn.recv()
            except (EOFError, OSError):
                self._drop(clients, client)
                return
            if writes:
                self._writes.append((client, writes))
            if method == "journal_read":
                cursors, timeout, limit = args
                deadline = None if timeout is None else time.monotonic() + timeout
                client.journal = _JournalWait(cursors, limit, deadline)
            if method is not None:  # a message with no method needs no reply
                client.pending = (method, args)
            try:
                if not client.conn.poll(0):
                    return
            except (EOFError, OSError):
                return

    @staticmethod
    def _reply(client: _Client, ok: bool, value: Any) -> None:
        client.pending = client.journal = None
        try:
            try:
                client.conn.send((ok, value))
            except Exception:
                if ok:
                    raise
                client.conn.send((False, RuntimeError(repr(value))))
        except OSError:
            pass  # the client has gone, and the next recv() will notice

    def _answer(self, clients: dict[Connection, _Client]) -> None:
        """Answer every request. Queued writes have been applied already."""
        readers = []
        for client in clients.values():
            if client.pending is None:
                continue
            method, args = client.pending
            if client.journal is not None:
                self._answer_journal(client)
            elif method == "read_many":
                readers.append((client, args[0]))
            else:
                try:
                    self._reply(client, True, self._dispatch(client, method, args))
                except Exception as err:
                    self._reply(client, False, err)
        if not readers:
            return
        try:
            results = self.db.read_many([op for _, ops in readers for op in ops])
        except Exception:
            # Read for each client separately, to find whose read failed.
            for client, ops in readers:
                try:
                    self._reply(client, True, self.db.read_many(ops))
                except Exception as err:
                    self._reply(client, False, err)
            return
        start = 0
        for client, ops in readers:
            self._reply(client, True, results[start : start + len(ops)])
            start += len(ops)

    def _answer_journal(self, client: _Client) -> None:
        waiting = client.journal
        assert waiting is not None
        now = time.monotonic()
        buffer = self._hub.buffer
        # Read only when the buffer has changed, or every few seconds. Reading
        # keeps the cursors, and the buffer's partitions, from expiring.
        if buffer.seq != waiting.seen or now >= waiting.next_read:
            waiting.seen, waiting.next_read = buffer.seq, now + 10
            room = (
                None if waiting.limit is None else waiting.limit - len(waiting.changes)
            )
            try:
                changes, waiting.cursors = buffer.read(waiting.cursors, 0, room)
            except Exception as err:
                self._reply(client, False, err)
                return
            if changes:
                waiting.changes.extend(changes)
                if waiting.ready_at is None:
                    # Wait briefly for more changes, so that one reply has them all.
                    waiting.ready_at = now + self._linger
        full = waiting.limit is not None and len(waiting.changes) >= waiting.limit
        ready = waiting.ready_at is not None and (now >= waiting.ready_at or full)
        expired = waiting.deadline is not None and now >= waiting.deadline
        if ready or expired:
            self._reply(client, True, (waiting.changes, waiting.cursors))

    def _dispatch(self, client: _Client, method: str, args: tuple) -> Any:
        if method == "write_many":
            ops, atomic = args
            return self.db.write_many(ops, atomic=atomic)
        if method == "journal_head":
            return self._hub.head(args[0])
        if method == "flush":
            self.db.flush(args[0])
            if client.errors:
                error = client.errors[0]
                client.errors.clear()
                raise error
            return None
        if method == "current_time":
            return self.db.current_time()
        if method == "capabilities":
            return self.db.capabilities
        raise ValueError(f"unknown method {method!r}")
