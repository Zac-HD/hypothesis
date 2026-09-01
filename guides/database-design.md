# Design: a database layer for distributed fuzzing

Status: draft for discussion, 2026-09-01.
Scope: the interface and backends in `hypothesis.database`, and the HypoFuzz storage schema built on them.
Prototype code and benchmarks accompany this document; results are in section 13.

## Summary

- `ExampleDatabase` grows three primitives beyond today's `key -> set[bytes]`:
  **maps** (`key -> {field: value}`), append-only **logs**, and a per-partition change **journal**.
  Keys and fields are tuples. The first component of a key is its **partition**:
  the unit of atomic batches, of journal subscriptions, and of physical placement.
- Existing subclasses keep working unchanged. The base class emulates the new methods
  on top of `save`, `fetch`, `delete` and change listeners. Native backends implement
  them directly and report what they guarantee through `capabilities`.
- Backends: in-memory (the reference), SQLite, Redis, and Postgres are native.
  The directory and GitHub-artifact databases run emulated.
- One process per machine owns the backend connection and serves it to that machine's
  worker processes over a local socket, so workers can be recycled freely.
- There is no coordinator. HypoFuzz machines coordinate through shared records:
  heartbeats, leases taken with compare-and-set, and a cluster index that every machine
  mirrors through the journal.
- Existing databases keep their failures with no migration step. A read-through wrapper
  copies failures and covering examples into a different store lazily, and HypoFuzz
  discards everything else it stored, once.

## 1. Where we are

Hypothesis stores `key -> set[bytes]` with `save`, `fetch`, `delete`, `move`, plus an
optional listener that receives every change in the database.
HypoFuzz builds everything else on top of that:

| HypoFuzz concept | How it is stored today | Consequence |
|---|---|---|
| latest progress report | save the new report, delete the previous one | two writes and two events per update |
| rolling observations | append, then trim to 300 on the client | the client keeps a shadow buffer per test |
| observation for a corpus entry | one key per choice sequence (`sha1` in the key) | loading a corpus costs one round trip per entry |
| failure state | three parallel keys, moved by delete plus save | not atomic; code deletes from both keys "just in case" |
| single records (fatal failure, worker identity) | fetch everything, delete each, save | racy read-modify-write |
| change notification | every process receives every change and filters by key prefix | cost grows with the square of the worker count |

These have concrete costs at cluster scale:

- **Event fan-out grows with the square of the worker count.** With N workers writing
  about 2 events per second each, every subscriber receives 2N events per second.
  At N = 100 that is 20,000 messages per second out of Redis, and about 200 JSON
  decodes per second on each worker. At N = 1,000 it is 2 million messages per second.
  Workers need only corpus and failure changes for the few tests they run.
- **Loading is one round trip per corpus entry.** A dashboard loading 1,000 tests with
  1,000 corpus entries each needs a million round trips: about 8 minutes at 0.5 ms,
  and hours across regions.
- **Reports grow without bound** and the dashboard loads all of them into memory.
- **Events are lossy.** The directory database reports deletions without the value,
  the dashboard drops events that arrive while it loads, and Redis pub/sub drops
  everything sent during a reconnect.
- **`RedisExampleDatabase` listeners never fire against a real server.**
  `_start_listening` registers a handler, but redis-py dispatches handlers only from
  `get_message()`, `listen()` or `run_in_thread()`, and nothing calls those.
  The tests pass because they pump `get_message()` by hand. Verified against Redis 7.0
  while writing this document.
- **The hub is single-machine.** Work assignment goes through `multiprocessing.Manager`,
  so nothing assigns tests, shares estimates, or tracks liveness across machines.

## 2. Requirements

Scale target, from the design discussion:

- 100 machines, each running 1 to 128 worker processes, so up to 12,800 workers.
- 10,000 tests.
- One store shared across repositories, branches, and versions over time.
- Amortized startup: a machine collects and subscribes to a subset of tests, chosen from
  the shared state.
- All coordination goes through the database. There is no separate service.
- Worker processes may be recycled at any time, because native code can corrupt a process.

Design constraints that follow:

1. **Subscriptions must be per partition.** A consumer must never receive changes for
   partitions it did not ask for. This removes the quadratic fan-out.
2. **One connection per machine.** Workers reach the store through a local server process,
   so every operation must be a plain request with byte payloads, and the change feed must
   be pull-based so the server can hold one subscription per partition for the whole machine.
3. **Conditional writes and a shared clock.** Leases and heartbeats need compare-and-set
   and a clock that all machines agree on. Expiry is optional in backends, so anything
   that needs expiry for correctness stores a deadline in the value and readers compare
   it with the store's clock.
4. **Whole collections in one round trip.** A test's corpus, with observations, loads in one read.
5. **Structure the store can index, but no schema.** Keys and fields have visible
   components; values stay opaque bytes.

## 3. Data model

### Keys, fields, partitions

A key is a non-empty tuple of components, each `bytes`, `str`, or `int`.
A field is a tuple of components, possibly empty.
A bare `bytes`, `str` or `int` is shorthand for a 1-tuple.

The **partition** of a key is its first component. Everything in one partition:

- can be written atomically in one batch,
- appears in one journal, in commit order,
- lives together physically: one Redis hash slot, one contiguous index range.

Namespaces (a project, or a shared store's tenant) are backend configuration, such as a
key prefix or a `namespace` column. They are not part of keys, so core Hypothesis does not
need to know about them. A namespace is also a natural sharding boundary.

### Legacy keys are sets

A key whose only component is `bytes` is a **legacy key**. Legacy keys hold sets: each
member is a field with one `bytes` component and an empty value. Any other use of a legacy
key raises `InvalidArgument`, in every backend.

The old methods act on legacy keys:

| old call | equivalent |
|---|---|
| `save(k, v)` | `map_put((k,), (v,), b"")` |
| `fetch(k)` | the fields of `map_items((k,))` |
| `delete(k, v)` | `map_delete((k,), (v,))` |

Core Hypothesis keeps its keys as they are: `dbkey`, `dbkey + b".secondary"`, and
`dbkey + b".pareto"`. Existing directory and Redis databases therefore remain valid,
byte for byte. Two consequences:

- The secondary and pareto keys are their own partitions. Nothing needs them to be
  atomic with the primary key, because HypoFuzz records failure state in its own map.
- HypoFuzz per-test data such as `(dbkey, "corpus")` shares the partition `dbkey` with
  core's primary failures, so one journal subscription covers a test.

### Encoding

Components are encoded as a type byte, then a length-prefixed payload:

| type | tag | payload |
|---|---|---|
| bytes | `0x01` | ULEB128 length, raw bytes |
| str | `0x02` | ULEB128 length, UTF-8 (`surrogatepass`) |
| int | `0x03` | 8 bytes, big-endian, sign bit flipped (int64 range) |

A tuple encodes as the concatenation of its components. The encoding is unambiguous,
and the encoding of a tuple is a byte prefix of the encoding of any tuple that extends it,
so prefix queries work on encoded bytes. `bool` encodes as `int`.

Ordering is not part of the contract, except that log entry ids compare as bytes.
An order-preserving encoding would need escaping, and choice sequences are full of zero bytes.

## 4. Interface

All of this lives on `ExampleDatabase`. The methods are synchronous and thread-safe.
An asynchronous wrapper runs them in worker threads, which is adequate because the load
per machine is hundreds of operations per second, not tens of thousands.

```python
KeyT = bytes | str | int | tuple[bytes | str | int, ...]
FieldT = KeyT  # may also be the empty tuple


class ExampleDatabase:
    # The old interface, unchanged. Subclasses implement save, fetch, delete.
    def save(self, key: bytes, value: bytes) -> None: ...
    def fetch(self, key: bytes) -> Iterable[bytes]: ...
    def delete(self, key: bytes, value: bytes) -> None: ...
    def move(self, src: bytes, dest: bytes, value: bytes) -> None: ...
    def add_listener(self, f: ListenerT) -> None: ...  # and the other listener methods

    # Maps: key -> {field: value}
    def map_get(self, key: KeyT, field: FieldT) -> bytes | None: ...
    def map_items(self, key: KeyT, *, prefix: FieldT = ()) -> dict[tuple, bytes]: ...
    def map_put(
        self,
        key: KeyT,
        field: FieldT,
        value: bytes = b"",
        *,
        ttl: float | timedelta | None = None,
        expect: bytes | None | Unset = unset,
    ) -> bool | None: ...
    def map_delete(
        self, key: KeyT, field: FieldT, *, expect: bytes | Unset = unset
    ) -> bool | None: ...
    def map_clear(self, key: KeyT) -> None: ...

    # Logs: append-only, ordered by the store, capped
    def log_append(
        self,
        key: KeyT,
        value: bytes,
        *,
        maxlen: int | None = None,
        ttl: float | timedelta | None = None,
    ) -> bytes | None: ...
    def log_range(
        self,
        key: KeyT,
        *,
        after: bytes | None = None,
        before: bytes | None = None,
        limit: int | None = None,
        reverse: bool = False,
    ) -> list[tuple[bytes, bytes]]: ...
    def log_trim(
        self,
        key: KeyT,
        *,
        maxlen: int | None = None,
        before: bytes | None = None,
        ids: Iterable[bytes] = (),
    ) -> int | None: ...

    # Batches. The single-operation methods above are sugar for these.
    def read_many(self, ops: Sequence[MapGet | MapItems | LogRange]) -> list[Any]: ...
    def write_many(
        self, ops: Sequence[WriteOp], *, atomic: bool = False
    ) -> list[Any]: ...

    # Change journal
    def journal_head(self, partition: bytes | str | int) -> bytes: ...
    def journal_read(
        self,
        cursors: Mapping[bytes | str | int, bytes],
        *,
        timeout: float | None = 0,
        limit: int | None = None,
    ) -> tuple[list[Change], dict[bytes | str | int, bytes]]: ...

    # Everything else
    capabilities: frozenset[str]

    def current_time(self) -> float: ...
    def flush(self, timeout: float | None = None) -> None: ...
    def close(self) -> None: ...  # also a context manager
```

`WriteOp` is one of the dataclasses `MapPut`, `MapDelete`, `MapClear`, `LogAppend`, and
`LogTrim`, with the same fields as the methods. The read operations are `MapGet`, `MapItems`,
and `LogRange`.

A backend author implements `read_many`, `write_many`, `capabilities`, and the three
journal hooks in section 4.4. Everything else has a default. A backend that holds
connections or threads also extends `close()`, which releases them. Using a closed database
opens what it needs again.

### 4.1 Maps

- `map_put` returns `True` if it applied the write, and `False` if a condition failed.
  A database that queues writes, such as the local server's client, returns `None`
  for a queued write. Only unconditional writes are queued.
- `expect=None` applies the write only if the field is absent. `expect=b"..."` applies it
  only if the current value equals those bytes. Together these give create-if-absent and
  compare-and-set.
- `ttl` says the entry may disappear any time after `ttl` seconds. It may also live forever.
  The most recent put sets the expiry, so a put with `ttl=None` removes an earlier one.
- `map_items` returns fields in an unspecified order. `prefix` restricts the result to fields
  that extend the given tuple.
- `map_delete` returns `True` if the field existed, and matched `expect` if one was given,
  or `None` if the write was queued.

### 4.2 Logs

- Entry ids are 16 bytes: 8 bytes of milliseconds on the store's clock, then an 8-byte
  sequence number. Ids strictly increase within a log, in the order the store applied the
  appends, so `log_range(after=last_seen)` paginates.
- `maxlen` keeps at least the newest `maxlen` entries and removes older ones eventually.
  Trimming is approximate so that backends can trim in bulk.
- `ttl` applies to the whole log: the `ttl` of the most recent append says entries older
  than that may be removed.
- `log_trim` is exact. It removes the entries in `ids`, and the entries before `before`,
  and then all but the newest `maxlen`. It returns the number of entries removed, or `None`
  if queued.
- `log_append` returns the new entry's id, or `None` if the database applied the write
  asynchronously, as the write-behind wrappers do.

### 4.3 Batches

- `read_many` runs its reads in one round trip where the backend allows it. It makes no
  promise that the reads see one snapshot.
- `write_many` applies its writes in order. With `atomic=True`, all operations must be in
  one partition, and either all apply or none do. Conditional operations are checked first,
  and one failed condition means nothing applies.
- Results line up with the operations: `bool` for puts and deletes, an id or `None` for
  appends, and the number of removed entries for trims.

### 4.4 Journal

Every change to a map or a log is recorded in its partition's journal, in the same
transaction as the change.

```python
class Change:  # immutable
    op: Literal["put", "delete", "clear", "append", "invalidate"]
    key: tuple  # key[0] is the partition
    field: tuple | None = None  # put, delete, and invalidate; None means the whole key
    entry_id: bytes | None = None  # append, and delete of a log entry
    value: bytes | None = (
        None  # put, append, and delete of a log entry; None if omitted
    )
```

Guarantees:

- **At least once.** A consumer may see a change twice, so applying changes must be idempotent.
- **Ordered within a partition, and unordered across partitions.**
- **Resumable.** `journal_read` returns new cursors, and any of them can be used again later.
- **Bounded retention.** The journal is for liveness, not history, and keeps entries for
  a few minutes (`journal_retention`, default 5 minutes). Each cursor records when it was
  issued. A read that reaches the end of a partition returns a new cursor, and a read that
  `limit` cut short keeps the old time. A cursor older than half the retention period is
  expired. The client checks this against its own clock, so backends need no bookkeeping.
  An expired cursor raises `JournalCursorExpired`, which names the partitions. The consumer
  then reloads those partitions and continues from `journal_head`.
- **Values inline when small**, up to 64 KiB by default. Larger values arrive as `None`.
- Trimming by `maxlen` or `before` is not journaled, and neither is expiry. Deleting log
  entries by id is journaled, because a reader cannot work it out. The change is a
  `delete` with `entry_id` set, and it carries the entry's value, so that the old listener
  API can name the member that went.
- `invalidate` means "something here changed, read it again". Emulated backends emit it
  when their change listener cannot say what changed.

The consumer loop is: take `journal_head(p)`, load the partition, then call
`journal_read` in a loop. Taking the head before the load means nothing is missed.
Anything that arrived during the load is applied twice, which is harmless.

`timeout=0` polls. `timeout=None` blocks until something arrives. A positive timeout
blocks for at most that long. Backends with the `blocking` capability wake up when a change
arrives. Others poll.

Backends implement three hooks, and one function in the base class does the rest: it
checks the age of each cursor, issues new ones, and handles the timeout. A position is
bytes that only the backend understands.

- `_journal_position(partition)` returns the position at the end of the partition.
- `_journal_fetch(positions, limit)` returns the changes after each position, the new
  positions, and the partitions that `limit` cut short.
- `_journal_wait(batch, timeout)` returns when a change may have arrived since that fetch,
  or when the timeout passes. Returning early does no harm.

### 4.5 Time, expiry, capabilities

- `current_time()` returns the store's clock in seconds, or the local clock if the store
  has none. Leases and heartbeats compare deadlines against this clock. Clients should
  measure the offset once a minute rather than call it on every check.
- `flush()` waits until every asynchronous write has been applied.
- `capabilities` is a subset of:

| capability | meaning |
|---|---|
| `native` | maps, logs, and batches are implemented natively, not emulated |
| `atomic` | atomic batches and `expect=` are really atomic |
| `journal` | the journal includes changes made by other processes and machines |
| `blocking` | `journal_read` with a timeout wakes up when a change arrives |
| `shared` | data is shared between machines |
| `ttl` | expired entries really disappear |
| `server_time` | `current_time()` is a clock shared by all clients |

HypoFuzz warns when more than one machine is running and the database lacks any of
`native`, `atomic`, `journal`, or `shared`. It warns again if journal lag or the write-behind
queue grows past a threshold. A measurement is more useful than a guess about scale.

### 4.6 Emulation, and the old methods as a view

In a database that implements only `save`, `fetch` and `delete`, the base class stores
structured data under reserved legacy keys:

| data | legacy key | members |
|---|---|---|
| map field | `MAGIC + b"m" + uleb(len(ek)) + ek + ef` | the value; normally one member |
| map index | `MAGIC + b"i" + ek` | `ef` for each field |
| log | `MAGIC + b"l" + ek` | 16-byte id followed by the value |

Here `ek` and `ef` are the encoded key and field, and `MAGIC` is `b"\x00hypothesis\x01"`.

Native backends interpret these reserved keys when they receive them through the old
methods. A user's wrapper that forwards only `save`, `fetch` and `delete` therefore sees the
same data as the wrapped database. It is slower and not atomic, but it is consistent.

Costs of emulation:

- `map_put` reads one small key. `map_items` reads the index, then each field.
- Concurrent puts to one field can leave two members. Reads pick the larger, and the next
  put removes the other.
- `expect=` checks and then writes, so it is not atomic.
- Expiry is not enforced.
- Log trimming is deferred until a quarter of `maxlen` has been appended by this process.
- The journal comes from the change listener. It exists only while this process has an
  open cursor, and it covers only what the listener reports.

## 5. Backends

### In-memory (the reference)

Dictionaries and lists behind one lock. Journal entries are kept only for partitions with
an open cursor, because no other process can hold one. Legacy listeners are called
synchronously, as they are today. The conformance tests compare every other backend with
this one.

### Directory (emulated)

Unchanged. The emulation costs one directory per map field, which is acceptable for local
use. The journal comes from watchdog, so deletions arrive as `invalidate`.

### SQLite

- Tables: `maps(key_hash, field_hash, key, field, value, expires_at)`,
  `logs(key_hash, id, value)`, `log_heads(key_hash, last_id, expires_at)`,
  and `journal(id INTEGER PRIMARY KEY, at, part_hash, op, key, field, entry_id, value)`.
- Primary keys use 16-byte BLAKE2b hashes of the encoded key and field, because choice
  sequences can be megabytes long. Postgres refuses index entries larger than about 2.7 kB.
- SQLite runs one writer at a time, so journal ids are assigned in commit order.
- WAL mode, `synchronous=NORMAL`, and one connection per thread.
- Other processes' commits are detected by polling `PRAGMA data_version`, which costs
  microseconds. Within one process, a condition variable wakes readers.
- Expired rows are filtered on read and deleted every few seconds.

WAL mode does not work on network filesystems, which matters for the choice of a default
(section 13).

### Redis (6.2 or later)

- Legacy keys stay as sets at `prefix + key`, exactly as `RedisExampleDatabase` stores
  them today, including the `expire_after` refresh. Existing data remains valid.
- A map is a hash at `prefix + "{" + tag + "}m" + ek`, and a log is a stream at
  `... "}l" + ek`. Here `tag` is the URL-safe base64 of the encoded partition, so every key
  of a partition lands in one cluster slot.
- Per-field expiry is a second hash of deadlines, filtered and cleaned inside the same
  script. A key-level `EXPIRE` set to the longest deadline removes idle maps entirely.
  Redis 7.4's `HEXPIRE` could replace this later.
- Reads of maps also run as scripts, so they compare deadlines with the server's clock,
  as writes do. Comparing with the client's clock would make an entry look alive to a
  read but expired to a conditional write, whenever the clocks disagreed.
- Each write is one Lua script. It applies the change, appends to the partition's journal
  stream with `XADD ... MINID ~ <now - retention>`, and publishes a wake-up on the
  partition's channel.
- Scripts are called by hash. A pipeline of redis-py `Script` objects checks that they
  exist first, which cost a second round trip on every call. If the server loses its
  scripts, as in a restart, every write in a batch fails, so the batch is sent again.
  Atomic batches send each script's text instead, because an unknown hash would fail
  inside `MULTI`, after the writes before it had been applied.
- `journal_read` subscribes to the wake-up channels of the partitions it follows. It reads
  with `XREAD` only the partitions that were woken, and sweeps all of them every few
  seconds in case a wake-up was lost.
- The old firehose channel (`listener_channel`) is still published, so the old listener API
  works. This version pumps messages in a thread, which fixes the bug above. The scripts
  build each message: a tag, then the legacy key with its length, then the value. Lua
  cannot encode base64, and the JSON that older versions sent doubled the size of every
  value. New listeners still parse that JSON. Old listeners never read the channel, because
  of the bug, so the change breaks nothing.
- Not done yet: sending each change in its wake-up message. A reader that stays connected
  would then need `XREAD` only after a reconnect.
- Not supported yet: Redis Cluster, because legacy keys are not hash-tagged.

### Postgres (13 or later)

- The same tables as SQLite, plus a `namespace` column.
- Writes to one partition are one statement: a call to a PL/pgSQL function, which takes an
  advisory lock for the partition, applies the operations, and calls `pg_notify` if it wrote
  to the journal. In
  autocommit mode a statement is a transaction. psycopg costs about 50 µs per statement
  on the test machine, which is why the statements are combined.
- Journal rows carry `xid8 DEFAULT pg_current_xact_id()` and a `bigserial`.
  A reader takes only rows with `xid < pg_snapshot_xmin(pg_current_snapshot())`, ordered
  by `(xid, id)`. Cursors are `(xid, id)`. Two facts make this correct:
  1. Every transaction older than `xmin` has finished, so the readable rows only ever grow
     at the end. This avoids the well-known gap where a serial id becomes visible after
     larger ones.
  2. The advisory lock is taken before a transaction's first write, so the transactions
     that write to one partition get transaction ids in the order they write. Without the
     lock, a transaction could get a small id from an early write, then write to a key
     after a later transaction had, and readers would apply the two writes in the wrong
     order. The conformance tests use one writer, so they cannot catch this. It came from
     working through the design.
- The lock also makes conditions safe to check with plain reads.
- The cost: a long-running writing transaction anywhere in the database delays delivery
  until it ends. It loses nothing. Read-only transactions do not delay it. Both facts were
  checked on Postgres 16.
- Log ids come from a per-log counter row (`log_meta`). Only appends to the same log
  contend for it.
- Commits are asynchronous by default (`synchronous_commit = off`, per transaction).
  A crash can lose the last fraction of a second of writes, but never corrupts anything,
  and this data is a cache. `synchronous_commit=True` turns this off.
- Wake-ups use `pg_notify` on one of 64 channels, chosen by a hash of the partition.
  Notifications carry no payload, because they are capped at 8 kB and lost on disconnect.
- `LISTEN` needs a session, not a pooled transaction. That is one connection per machine.
- Expired rows are filtered on read and deleted in the background.

### Other stores

- **DynamoDB** fits well: partition key = partition, sort key = the rest, transactions of
  up to 100 items, and a journal made of items with a counter item per partition.
  Reading the journal would be a query loop.
- **Any old-style database** works through the emulation, with the costs listed in 4.6.

### Wrappers

- `ReadOnlyDatabase` forwards reads and journal calls, and ignores writes.
- `MultiplexedDatabase` merges reads: the union for maps, and a merge by id for logs.
  It writes to every database, and combines cursors. Atomic batches are atomic in each
  database separately.
- `BackgroundWriteDatabase` queues writes that are neither atomic nor conditional, and
  applies reads after the queue drains. Queued writes return `None`.
- `ReadThroughDatabase(primary, fallback)` is new, and is described in section 9.
- Every wrapper forwards `close()` to the databases it wraps.

## 6. The per-machine database server

`hypothesis fuzz` starts a parent process that opens the real backend and serves it on a
local socket: a Unix socket, or a named pipe on Windows (untested), authenticated with a random key
through `multiprocessing.connection`. Worker processes receive
`settings.database = RemoteDatabase(address, authkey)`.

The parent never runs test code, so a crashing worker cannot take the connection down.

Guarantees to each client connection:

- Writes are queued and the call returns immediately, unless the write is atomic or
  conditional.
- Writes from one client are applied in the order sent.
- A read sees that client's earlier writes. Queued writes travel with the client's next
  request, and the server applies them before it answers.

How the server works:

- One thread reads every connection. Each time round, it applies the writes it has
  received as one `write_many`, then answers the reads of every client as one
  `read_many`. For Redis and Postgres, that is one round trip for many clients.
- While that thread is busy, requests wait in the sockets. When the backend is slow,
  clients block, and nothing queues up without limit.
- Each pass costs little when nothing happens. Connections stay registered with one
  selector, and a client that waits for changes is checked only when the buffer has
  changed. Before those two fixes, four idle followers cut another client's reads from
  5,300 to 1,000 a second, and 64 idle connections cut them to 3,200.
- A second thread follows the backend's journal, with one subscription per partition for
  the whole machine, and keeps the changes in memory. Clients read from there.
- The server issues its own cursors. They name the server instance, so a client that
  outlives a server restart gets `JournalCursorExpired` and reloads.
- Not yet: caching reads for partitions it already follows, so that a recycled worker
  reloads its corpus from memory.

Details that the benchmarks decided:

- The first version had a thread per connection. Every request then passed the GIL
  between threads, and each handoff cost a futex call and a context switch. With 16
  clients and SQLite, the server spent 1.9 ms of CPU per read. With one loop thread it
  spends 140 µs, and it serves 8.5 times as many reads.
- A separate writer thread had the same problem, on a smaller scale. A read that followed
  a write waited for two handoffs.
- The client sends queued writes after 5 ms, or 200 operations, or with its next request,
  whichever comes first. Sending each write as its own message cost about 80 µs of server
  CPU per write, half of it in system calls.
- One thread per client process sends the delayed batches. Starting a `threading.Timer`
  for each batch cost 271 µs, more than the batch.
- After a journal read that returned changes, the server waits 5 ms before reading again,
  so that one read collects several changes. Without this, the server woke for every
  change, and journal lag reached seconds under load.
- A round trip costs about 150 µs on the test machine, and `multiprocessing.connection`
  accounts for about 100 µs of that. A bare pipe takes 41 µs. A protocol that reads each
  message with one system call would help.

Workers are started with `spawn` and import only the test files for their assigned tests.
Reloading a worker takes seconds, with no collection step.

## 7. Core Hypothesis

- At the start of a test, the reuse phase reads the primary, secondary, and pareto keys
  with one `read_many`. That is one round trip instead of three. Over a remote store with
  a 30 ms round trip, a suite of 5,000 tests saves about five minutes.
- Writes stay synchronous. Writing in the background during shrinking is future work.
- Keys and values do not change, so plain Hypothesis and HypoFuzz still see each other's
  failures.

## 8. HypoFuzz schema

Values are JSON, so they can be read in `psql` and `redis-cli`. The one exception is
noted below.

### 8.1 Per test: partition `T`, the test's database key

| key | kind | contents |
|---|---|---|
| `(T,)` | set | shrunk failures (core Hypothesis's key) |
| `(T + b".secondary",)` | set | unshrunk failures (core Hypothesis's key; its own partition) |
| `(T, "failure-info")` | map | field: the choice sequence. Value: state, origin, observation, timestamps |
| `(T, "corpus")` | map | field: the choice sequence. Value: the observation, or empty |
| `(T, "reports")` | log | reports at coverage changes and phase changes; `ttl` 30 days, `maxlen` 2,000 (section 11) |
| `(T, "history")` | map | at most 1,000 coarse points, kept indefinitely (8.5) |
| `(T, "progress")` | map | field: the worker. Value: its latest timed report; `ttl` 1 hour |
| `(T, "campaigns")` | map | field: the worker. Value: cumulative counts for that campaign |
| `(T, "observations")` | log | rolling observations, `maxlen` 300 |
| `(T, "watchers")` | map | field: a dashboard or collector. Value: requested rate and time; `ttl` 60 seconds |
| `(T, "leases")` | map | for example `("shrink", origin)`. Value: holder and deadline |
| `(T, "meta")` | map | `("nodeid",)`, `("function",)`, `("fatal",)`, `("migrated",)` |

### 8.2 The cluster index: partition `"index"`

| key | kind | contents |
|---|---|---|
| `("index", "tests")` | map | field: `(T,)`. Value: a summary with nodeid, behaviors, corpus size, failure state, estimated behaviors per second, last activity |
| `("index", "names")` | map | fields `("nodeid", n)` and `("function", f)`. Value: recent database keys for that name |
| `("index", "machines")` | map | field `(m,)`: a heartbeat every 15 seconds. Field `(m, "assignments")`: the machine's tests, packed as binary pairs of an 8-byte key prefix and a worker count, written at most once a minute |
| `("index", "announce")` | log | new failures, fatal errors, new tests; `maxlen` 10,000 |
| `("index", "meta")` | map | `("schema",)`: the schema version |

A string component cannot collide with a `bytes` database key, so `"index"` is safe.

A separate partition, `("collection", commit)`, maps each nodeid to its database key and
file. The first machine to start on a commit publishes it, under the lease
`("collect", commit)`. Other machines read it once and import only the files they need.
It lives outside the index partition so that publishing it does not flood every mirror.

### 8.3 Failures

Goals: never shrink the same failure twice, drop unshrunk entries quickly, and keep a
failure until it has been seen passing for a while.

1. **Found.** A worker finds origin `O` with choices `C`. If the mirrored `failure-info`
   already has an entry for `O`, it does nothing. Otherwise it writes `C` to the secondary
   key and to `failure-info` as unshrunk, then tries to take the lease `("shrink", O)`.
2. **Shrinking.** The lease holder shrinks and renews the lease as it goes. Other workers
   see the lease and leave `O` alone. If the holder dies, the lease expires, and the next
   worker to load the test resumes from the unshrunk entry.
3. **Shrunk.** The holder writes `C_min` to the primary key and to `failure-info` as
   shrunk. It deletes every unshrunk entry for `O` and releases the lease.
4. **Replay.** A worker starting the test replays every entry in `failure-info`.
   If the entry reproduces, the worker clears `passing_since`.
   If it does not, the worker sets `passing_since`, unless it is already set.
   An entry whose `passing_since` is older than `failure_ttl` is deleted from all three keys.
   The default `failure_ttl` is 8 days, and it is configurable.
5. **Plain Hypothesis** still deletes a failure from the primary key when it does not
   reproduce. HypoFuzz treats `failure-info` as the source of truth, and writes the primary
   entry again if the failure reproduces.

This replaces the shrunk, unshrunk, and fixed keys with a single record.
Branches that share a namespace can disagree about whether a failure reproduces.
That flapping is limited to `passing_since`. Users who care should use a namespace per branch.

### 8.4 Corpus

- Loading is one `map_items((T, "corpus"))`, with observations included.
- Evicting an entry is one `map_delete`.
- Open question, which does not affect the interface: should values also carry the
  coverage fingerprint? A new worker could then start mutating at once and replay the corpus
  gradually, instead of replaying it all before doing useful work.

### 8.5 Reports and history

- **Timed progress** overwrites `(T, "progress")[worker]`. It used to be an append and a
  delete.
- **Change points** append to `(T, "reports")` with a 30-day `ttl`. During replay, a worker
  writes one report when replay ends, not one per replayed entry. This matters for storage
  (section 10).
- **Coarse history:** each change point is also written to `(T, "history")`, with the
  report's log id as the field. When a writer sees more than 1,000 points, it takes the lease
  `("thin-history",)` and deletes every second point in the older half. Density then falls
  off roughly logarithmically with age. It is a map, not a log, because thinning deletes
  entries from the middle.
- **Cumulative totals** (inputs, elapsed time) come from `(T, "campaigns")`, one record per
  worker campaign. They no longer come from summing every report, so expiring old reports
  loses no totals.

### 8.6 Observations and watchers

- A dashboard viewing a test, or a collector, writes `(T, "watchers")[client]` with the rate
  it wants, every 10 to 20 seconds. The records have a 60-second `ttl`, and also carry the
  time they were written.
- Workers follow their tests' journals anyway, so they see watcher changes at no extra cost.
  A worker records observations at the highest requested rate (default: 1 per second),
  or at 1 per minute if nobody is watching.
- `hypothesis fuzz --collect-observations NODEID... --out FILE` registers as a watcher and
  writes observations to a JSON-lines file. It can watch every test, but at full scale that
  costs about 38 MB/s (section 10).

### 8.7 Liveness, leases, and scheduling

- **Leases** are map fields with the value `{holder, deadline}`. To acquire one,
  `map_put(expect=None)`. If that fails and the deadline has passed by `current_time()`,
  `map_put(expect=<the old bytes>)`. Renewing and releasing use `expect=<my bytes>`.
- **Machines** send heartbeats every 15 seconds. Readers consider a machine dead after
  60 seconds without one.
- **Scheduling** runs on each machine. Every 30 seconds, using its mirror of the index,
  a machine scores each test by its estimated value per second, divided by one plus the
  number of workers already on it across the cluster. Tests it already has loaded get a
  bonus. Random jitter breaks ties between machines. Then it publishes its assignments.
  This converges loosely. Putting a few extra workers on a test is harmless, and it
  already happens today.
- **Estimator state** for a new worker comes from the index summary, not from zero. This
  addresses the "lookback across workers" note in `bayes.py`.
- **Stopping.** When `failure-info` for a test has a shrunk entry, every machine drops the
  test's priority. Today, each worker stops only when it finds a failure itself.

### 8.8 Identity: database keys and nodeids

- The database key is the identity. `("index", "names")` records the last few database keys
  seen for each nodeid and each function name.
- When a worker starts a test whose partition is empty, it looks up the nodeid, or the
  function name if it has no nodeid, and finds the previous database key. It queues that
  key's corpus and failures for replay at low priority. Whatever still reproduces or still
  adds coverage is saved under the new key by the normal code paths.

### 8.9 Garbage collection

- Backends that enforce expiry clean up `ttl` entries by themselves.
- A test whose last activity is older than 90 days (configurable) gets its known keys
  cleared, by whichever machine holds the lease `("gc",)` in the index. This works in
  backends without expiry, and it removes partitions left behind when a test's code changed.
- Old-format HypoFuzz keys expire in Redis, through `expire_after`. In the directory
  database they remain until someone runs `hypothesis fuzz --cleanup-legacy`.

## 9. Migration

- **Same store, upgraded in place.** Core Hypothesis's keys do not change, and neither does
  their layout, so failures need no migration. On first start of a test, HypoFuzz reads the
  old corpus set `T + b".hypofuzz.corpus"`, which is an ordinary legacy key. It puts each
  entry into `(T, "corpus")` with an empty observation, then sets `(T, "meta")["migrated"]`.
  Everything else in the old format is ignored.
- **A different store.** `ReadThroughDatabase(primary, fallback)`. The first time a legacy
  key is read, it also reads the key from `fallback`, copies the members into `primary`, and
  writes the marker `(k, "_meta")["read-through"]` in the key's own partition. Later reads
  use only `primary`. This copies failures, secondary entries, and the pareto front without
  translating any keys. HypoFuzz's corpus step above runs through the same wrapper.
- Failure observations are not copied. HypoFuzz replays every failure anyway, and records a
  new observation.

## 10. Load and storage at full scale

Assumptions: 100 machines with 128 workers each, 10,000 tests, 200 corpus entries per test
on average, 3 KB per observation, 300 bytes per report, and workers that live about 4 hours.

Writes in steady state:

| stream | writes/s | bytes/s | notes |
|---|---|---|---|
| observations, 1 per minute unless watched | ~215 | 0.6 MB | 1 per second everywhere would be 12,800/s and 38 MB/s |
| timed progress | ~215 | 65 kB | overwrites |
| change-point reports | ~40 | 12 kB | 5 per campaign start, and about 9 campaign starts per second |
| corpus | ≲100 | ≲0.3 MB | a brand-new project writes about 3,300/s for its first 10 minutes |
| index summaries | ~215 | 65 kB | every machine receives all of it: about 6.5 MB/s out of the store |
| heartbeats and assignments | ~10 | 20 kB | |
| **total** | **~800** | **~1.2 MB** | the journal doubles the row writes |

That is well within one Postgres or one Redis, and the dominant costs are storage and the
per-machine mirrors.

Storage:

| data | estimate | notes |
|---|---|---|
| corpus with observations | 6.4 GB | 10,000 tests × 200 entries × 3.2 kB |
| observation logs | 9 GB | 10,000 × 300 × 3 kB. `maxlen=100` would make it 3 GB |
| coarse history | 3 GB | 10,000 × 1,000 × 300 bytes |
| fine-grained reports, 30 days | **35 GB, 80% interval 2–800 GB** | see below |
| journal | under 0.5 GB | 5 minutes of writes |

Fine-grained reports dominate, and the estimate is very uncertain. The reasoning:

- A campaign is one worker on one test. Every campaign starts by replaying the corpus.
- Today, every replayed entry that adds coverage writes a report. That is about 200 reports
  per campaign start, from replay alone.
- With one report at the end of replay, a campaign writes about 5 reports (80% interval: 2 to 30).
- Campaign starts per day are 12,800 workers × 10 tests each × (24 hours ÷ worker lifetime).
  With a 4-hour lifetime that is 770,000 a day. Lifetimes from 1 to 24 hours give 128,000 to
  3 million a day.
- 5 reports × 770,000 a day × 30 days × 300 bytes ≈ 35 GB. At today's 200 reports per
  campaign start it would be about 1.4 TB.

Conclusions:

1. Coalescing replay reports is required at this scale, not optional.
2. A 30-day `ttl` alone does not bound storage, because storage scales with the campaign
   start rate. So each test's report log is also capped, by `report_log_maxlen` in section
   11. The default of 2,000 bounds the total at about 6 GB.
3. Postgres suits full scale. Redis suits up to about 1,000 workers, or a deployment with
   30 to 60 GB of memory to spare.

## 11. Tunable controls

Each of these trades freshness or history against storage or load. Each has one name, one
default, and one place where it is set. In HypoFuzz, each is a named setting with a
command-line flag, and the dashboard shows the values in use. A change to a default that
bounds storage should come with a new estimate for section 10.

| control | default | set in | trades |
|---|---|---|---|
| `failure_ttl` | 8 days | HypoFuzz | how long a fixed failure is kept, against storage |
| `report_ttl` | 30 days | HypoFuzz | how far back the dashboard shows change points, against storage |
| `report_log_maxlen` | 2,000 per test | HypoFuzz | the same, for tests that restart often. Bounds reports at about 6 GB at full scale |
| `history_points` | 1,000 per test | HypoFuzz | detail in the coarse history, against 3 GB at full scale |
| `observation_log_maxlen` | 300 per test | HypoFuzz | observations the dashboard can show, against 9 GB at full scale |
| `observation_rate` | 1 per second if watched, else 1 per minute | HypoFuzz | freshness, against write load |
| `watcher_ttl` | 60 seconds | HypoFuzz | how long workers keep a closed dashboard's rate |
| `heartbeat_interval`, `machine_timeout` | 15 and 60 seconds | HypoFuzz | how soon a dead machine's tests move, against writes |
| `scheduling_interval` | 30 seconds | HypoFuzz | how soon work rebalances, against churn |
| `gc_inactivity` | 90 days | HypoFuzz | when an inactive test's data is deleted |
| `journal_retention` | 5 minutes | each database | how far a reader can fall behind before it reloads, against journal storage |
| `INLINE_VALUE_LIMIT` | 64 KiB | `hypothesis.database` | journal size, against re-reading large values |
| `batch_delay`, `batch_size` | 5 ms, 200 operations | `RemoteDatabase` | batching, against latency |
| `linger` | 5 ms | `DatabaseServer` | batching journal reads, against lag |
| `poll_timeout` | 0.5 seconds | `DatabaseServer` | how late a newly followed partition's first changes can be, against idle reads |
| `sweep_interval` | 5 seconds | Redis | how late a lost wake-up is noticed, against reads |
| `synchronous_commit` | off | Postgres | durability of the last fraction of a second, against write latency |
| `poll_interval` | 10 ms | SQLite | journal latency, against idle work |

## 12. Testing

- **Conformance:** one `RuleBasedStateMachine` runs against every backend, and against every
  wrapper around every backend. It checks maps, logs, batches, and conditions against a plain
  Python model, and checks that a journal mirror converges to the model.
- **Emulation:** a wrapper that forwards only the old methods must behave like the database
  it wraps, apart from atomicity and log ids.
- **Expiry:** timing-based unit tests, for backends with `ttl`.
- **Compatibility:** the existing listener tests keep passing, and existing directory and
  Redis data remains readable.
- **Real time:** the test suite fakes the clock, so that timing does not make tests flaky.
  These tests wait for other threads and processes, so they use the real clock instead, and
  the library needs no code that works around a fake one.

Status of the prototype: the state machine passes against the in-memory, SQLite, directory,
Redis, and Postgres backends, through `RemoteDatabase`, and through a wrapper that forwards
only the old methods. It ran 200 to 300 examples of 40 steps each. The existing database
tests pass unchanged, except that the listener tests now close their database at teardown.

## 13. Benchmarks

The code is in `hypothesis/benchmark/database/`, and its README describes the workloads.
The figures below come from one run of 8 seconds per configuration.

### The test machine, and what that means for the figures

- A virtual machine with 4 cores of a 2.8 GHz Xeon. Redis 7.0, Postgres 16, and every
  worker process ran on it together, so they competed for CPU.
- System calls are slow here: about 12 µs each, and a round trip through a pipe takes 41 µs.
- Runs of the same configuration differed by 10 to 20%. Differences smaller than about 25%
  are not meaningful.
- With 16 or 64 processes, the workers alone need more than the 4 cores. Those columns
  measure a machine that is overloaded, as well as the backend.

### Plain Hypothesis: reading a test's keys

Operations per second. A `core` operation is one `read_many` of three keys, or three
`fetch` calls one time in ten.

| backend | mode | 1 proc | 4 procs | 16 procs | 64 procs |
|---|---|---|---|---|---|
| memory | direct | 26,776 | | | |
| directory | direct | 3,793 | 8,213 | 8,034 | 7,686 |
| sqlite | direct | 8,167 | 15,762 | 15,173 | 13,194 |
| redis | direct | 2,670 | 8,570 | 9,763 | 6,484 |
| postgres | direct | 1,044 | 7,597 | 6,522 | 4,915 |
| sqlite | server | 2,959 | 4,468 | 5,422 | 5,499 |

Latency at 4 processes, p50 / p99, in µs:

| backend | three `fetch` calls | one `read_many` | `save` |
|---|---|---|---|
| directory | 437 / 1,028 | 452 / 1,031 | 368 / 959 |
| sqlite | 240 / 591 | 199 / 516 | 167 / 4,215 |
| redis | 720 / 1,746 | 319 / 947 | 463 / 1,137 |
| postgres | 627 / 1,449 | 433 / 948 | 721 / 1,364 |

- `read_many` halves the cost of reading a test's keys from Redis or Postgres, which is
  what section 7 predicted. Local backends gain little.
- Plain Hypothesis should connect directly. The local server adds a hop, and it is
  for fuzzing, where writes are frequent.

### HypoFuzz: writing

Operations per second, for the `fuzz` workload at full speed.

| backend | mode | 1 proc | 4 procs | 16 procs | 64 procs | server µs/op |
|---|---|---|---|---|---|---|
| memory | server | 44,834 | 78,442 | 70,698 | 62,977 | 12 |
| directory | direct | 3,152 | 6,609 | 2,039 | 2,043 | |
| directory | server | 2,750 | 2,863 | 1,860 | 2,598 | 353 |
| sqlite | direct | 3,188 | 5,297 | 4,789 | 2,367 | |
| sqlite | server | 9,732 | 9,500 | 9,926 | 7,862 | 71 |
| redis | direct | 2,426 | 6,851 | 10,186 | 7,174 | |
| redis | server | 10,985 | 10,243 | 10,134 | 8,636 | 55 |
| postgres | direct | 1,224 | 4,813 | 3,702 | 2,965 | |
| postgres | server | 5,036 | 5,000 | 5,092 | 4,783 | 98 |

- Through the local server, SQLite, Redis, and Postgres write 2 to 4 times as fast as one
  process writing directly. The server batches, so the number of client processes hardly
  matters.
- A queued write returns in 5 to 6 µs at the median, and 60 µs at p99.
- The design load in section 10 is about 10 writes per second per machine. Every backend
  has a hundred times that, on one machine. The central store's capacity is what matters.
- The directory database gains nothing from batching, because its emulation does file
  operations for each write.

### HypoFuzz: loading a corpus

Latency in µs, p50 / p99, to load 200 entries with 3 kB observations, at 4 processes.

| backend | mode | old layout, one `fetch` per entry | one `map_items` | speed-up |
|---|---|---|---|---|
| directory | direct | 9,700 / 13,793 | 10,205 / 14,856 | none |
| sqlite | direct | 7,447 / 11,110 | 1,315 / 1,994 | 5.7× |
| redis | direct | 59,163 / 94,294 | 4,480 / 7,935 | 13× |
| postgres | direct | 43,944 / 57,189 | 5,620 / 8,058 | 7.8× |
| sqlite | server | 105,956 / 145,161 | 3,454 / 6,120 | 31× |
| redis | server | 205,859 / 243,037 | 6,986 / 17,703 | 29× |
| postgres | server | 248,783 / 325,879 | 10,890 / 29,819 | 23× |

- Loading a corpus in one read is the largest gain in this design. Across a network, the
  old layout's cost grows with the round-trip time, and the new one's does not.
- The directory database's emulation also reads one file per entry, so it gains nothing.
- Redis is slower than SQLite here, probably because redis-py parses replies in Python.
  `hiredis` is not installed on the test machine.

### Journal lag

Milliseconds from a write to its arrival, at 2,000 writes per second, with one reader
that follows all 101 partitions. Every change arrived, and no cursor expired.

| backend | mode | procs | p50 | p99 | max |
|---|---|---|---|---|---|
| sqlite | direct | 4 | 5.2 | 14.7 | 90 |
| sqlite | direct | 16 | 5.7 | 40.4 | 189 |
| redis | direct | 4 | 1.4 | 4.0 | 61 |
| redis | direct | 16 | 1.9 | 10.5 | 48 |
| postgres | direct | 4 | 2.3 | 8.2 | 55 |
| postgres | direct | 16 | 3.6 | 40.1 | 233 |
| memory | server | 4 | 11.3 | 28.6 | 88 |
| sqlite | server | 4 | 17.4 | 107.6 | 166 |
| redis | server | 4 | 16.7 | 66.5 | 148 |
| postgres | server | 4 | 22.8 | 140.6 | 197 |

- Through the server, lag includes three batching windows of 5 ms: the client's, the
  journal thread's, and the reply's. That is a deliberate trade for throughput.
- A newly followed partition's first changes can arrive up to `poll_timeout` (0.5 s) late,
  because the journal thread reads the new partition only when its current read returns.
  A first version of the benchmark measured this by mistake, and reported a p99 of 450 ms.

### SQLite or the directory database?

The question was whether SQLite should replace the directory database as a default.

For plain Hypothesis:

- SQLite reads twice as fast: 8,200 against 3,800 operations per second in one process,
  and 15,800 against 8,200 in four.
- In absolute terms that is small. The difference is about 250 µs per test, so about
  1.3 seconds for a suite of 5,000 tests.
- SQLite's writes have a worse p99 when several processes write, because writers take
  turns. Plain Hypothesis writes rarely, so this matters little.
- SQLite needs a local filesystem, because its write-ahead log uses shared memory. The
  directory database works on network filesystems.
- Some users commit the directory database to version control. A SQLite file cannot be
  merged.

For HypoFuzz, behind the local server:

- SQLite writes 3.5 times as fast as the directory database: 9,700 against 2,800
  operations per second.
- SQLite's journal works, with a p50 lag of 5 ms. The directory database's journal comes
  from watchdog, whose tests are skipped as flaky.

Status: undecided. For HypoFuzz, the evidence favors SQLite. For plain Hypothesis, the gain
is real but small, and so are the compatibility costs. This evidence would settle it:

- How many projects commit `.hypothesis/examples` to version control, or keep it on a
  network filesystem. A code search for committed example directories would give a count.
- Whether the directory database's many small files cause trouble, such as slow CI caches
  or inode limits.
- The time per test on a large real suite, with each backend. The benchmark here is
  synthetic.

### Bugs the benchmarks found

- **Postgres:** when 64 processes connected at once, `CREATE INDEX IF NOT EXISTS` deadlocked
  with processes that were already writing. It locks its table even when the index exists.
  The schema statements now run only when something is missing.
- **SQLite:** one process failed with "database is locked" at `PRAGMA journal_mode=WAL`,
  while other processes opened the same new file. Three attempts to reproduce this in
  isolation failed. The fix reads the mode first, and retries. It is defensive.
- **The local server:** the first two designs cost more in thread handoffs than in work.
  Section 6 has the details.

## 14. Non-goals and open questions

Not in version 1:

- Content-addressed values.
- Counters.
- Transactions across partitions.
- Listing keys.
- A Redis client written for Trio.
- Redis Cluster.

Open questions:

1. Whether corpus values carry fingerprints (section 8.4).
2. Whether to make SQLite the default, for HypoFuzz and for plain Hypothesis. Section 13
   has the evidence so far, and what would settle it.
3. The batching windows in the local server. Three windows of 5 ms give 11 to 25 ms of
   journal lag. Halving them would halve the lag, and cost some batching.
4. A faster local transport. `multiprocessing.connection` costs about 100 µs per round
   trip on the test machine.
5. Windows. The local server should work with named pipes there, but nothing tests it.

Decided since the first draft:

- `close()` is part of the interface (section 4).
- Each test's report log is capped, by a tunable control (section 11).
