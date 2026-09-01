# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Benchmarks for the database backends. See README.md in this directory."""

import argparse
import hashlib
import json
import multiprocessing
import os
import random
import shutil
import statistics
import struct
import sys
import tempfile
import time
import uuid
from collections import defaultdict

from hypothesis.database import (
    DirectoryBasedExampleDatabase,
    InMemoryExampleDatabase,
    JournalCursorExpired,
    MapItems,
    SQLiteExampleDatabase,
    serve_database,
)

TESTS = 100
TESTS_PER_WORKER = 10
SAMPLES_PER_OP = 20_000


def test_key(i):
    return hashlib.sha384(b"test-%d" % i).digest()


def stamped(size):
    """A value of ``size`` bytes, starting with the current time."""
    return struct.pack(">d", time.time()) + os.urandom(max(0, size - 8))


def make_backend(spec):
    kind = spec["kind"]
    if kind == "memory":
        return InMemoryExampleDatabase()
    if kind == "directory":
        return DirectoryBasedExampleDatabase(spec["path"])
    if kind == "sqlite":
        return SQLiteExampleDatabase(spec["path"])
    if kind == "redis":
        from redis import Redis

        from hypothesis.extra.redis import RedisExampleDatabase

        return RedisExampleDatabase(
            Redis.from_url(spec["url"]),
            key_prefix=spec["prefix"].encode(),
            listener_channel="",
        )
    if kind == "postgres":
        from hypothesis.extra.postgres import PostgresExampleDatabase

        return PostgresExampleDatabase(spec["url"], namespace=spec["namespace"])
    raise ValueError(kind)


class Recorder:
    def __init__(self):
        self.samples = defaultdict(list)
        self.counts = defaultdict(int)

    def timed(self, name, fn, *args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        self.counts[name] += 1
        samples = self.samples[name]
        if len(samples) < SAMPLES_PER_OP:
            samples.append(elapsed)
        elif random.random() < SAMPLES_PER_OP / self.counts[name]:
            samples[random.randrange(SAMPLES_PER_OP)] = elapsed
        return result


# Workloads. Each runs until the deadline and returns the number of operations.


def fetch_each(db, keys):
    return [list(db.fetch(k)) for k in keys]


def core_workload(db, rec, deadline, rate):
    ops = 0
    while time.perf_counter() < deadline:
        key = test_key(random.randrange(TESTS))
        keys = [key, key + b".secondary", key + b".pareto"]
        if ops % 10 == 9:
            rec.timed("core: 3 fetch calls", fetch_each, db, keys)
        else:
            rec.timed(
                "core: read_many of 3 keys", db.read_many, [MapItems(k) for k in keys]
            )
        ops += 1
        if random.random() < 0.05:
            rec.timed("core: save", db.save, key + b".secondary", os.urandom(200))
            ops += 1
    return ops


def fuzz_workload(db, rec, deadline, rate):
    """Runs flat out, or at ``rate`` operations per second."""
    tests = random.sample(range(TESTS), TESTS_PER_WORKER)
    worker_id = uuid.uuid4().bytes
    ops = 0
    started = time.perf_counter()
    while time.perf_counter() < deadline:
        if rate and (ahead := ops / rate - (time.perf_counter() - started)) > 0:
            time.sleep(ahead)
        t = test_key(random.choice(tests))
        rec.timed(
            "fuzz: observation append",
            db.log_append,
            (t, "observations"),
            stamped(3000),
            maxlen=300,
        )
        rec.timed(
            "fuzz: progress put",
            db.map_put,
            (t, "progress"),
            worker_id,
            stamped(300),
            ttl=3600,
        )
        ops += 2
        if random.random() < 0.1:
            rec.timed(
                "fuzz: corpus put",
                db.map_put,
                (t, "corpus"),
                os.urandom(200),
                stamped(3000),
            )
            ops += 1
        if random.random() < 0.05:
            rec.timed(
                "fuzz: index put", db.map_put, ("index", "tests"), t, stamped(300)
            )
            ops += 1
        if random.random() < 0.02:
            rec.timed(
                "fuzz: report append",
                db.log_append,
                (t, "reports"),
                stamped(300),
                ttl=30 * 86400,
            )
            ops += 1
    return ops


STARTUP_TESTS = 20
CORPUS_SIZE = 200


def legacy_corpus_key(t):
    return t + b".hypofuzz.corpus"


def legacy_observation_key(t, choices):
    return t + b".hypofuzz.corpus." + hashlib.sha1(choices).digest() + b".observation"


def startup_setup(db):
    for i in range(STARTUP_TESTS):
        t = test_key(i)
        for _ in range(CORPUS_SIZE):
            choices, observation = os.urandom(200), os.urandom(3000)
            db.map_put((t, "corpus"), choices, observation)
            db.save(legacy_corpus_key(t), choices)
            db.save(legacy_observation_key(t, choices), observation)
    db.flush()


def load_legacy(db, t):
    """Load a corpus in the old layout, with one fetch per entry."""
    return [
        list(db.fetch(legacy_observation_key(t, choices)))
        for choices in db.fetch(legacy_corpus_key(t))
    ]


def startup_workload(db, rec, deadline, rate):
    ops = 0
    while time.perf_counter() < deadline:
        t = test_key(random.randrange(STARTUP_TESTS))
        if ops % 2:
            found = rec.timed("startup: one map_items", db.map_items, (t, "corpus"))
        else:
            found = rec.timed(
                "startup: legacy, one fetch per entry", load_legacy, db, t
            )
        assert len(found) == CORPUS_SIZE, len(found)
        ops += 1
    return ops


WORKLOADS = {"core": core_workload, "fuzz": fuzz_workload, "startup": startup_workload}


def setup(db, workload):
    if workload == "startup":
        startup_setup(db)
    elif workload == "core":
        for i in range(TESTS):
            for _ in range(3):
                db.save(test_key(i), os.urandom(200))


def run_worker(db_or_spec, workload, worker, barrier, seconds, rate, results):
    random.seed(worker)
    db = make_backend(db_or_spec) if isinstance(db_or_spec, dict) else db_or_spec
    if isinstance(db_or_spec, dict) and db_or_spec["kind"] == "memory":
        setup(db, workload)  # this process's database is not shared, so set it up here
    WORKLOADS[workload](db, Recorder(), time.perf_counter() + 0.5, rate)  # warm up
    rec = Recorder()
    db.flush()
    barrier.wait()
    start = time.perf_counter()
    ops = WORKLOADS[workload](db, rec, start + seconds, rate)
    db.flush()
    elapsed = time.perf_counter() - start
    results.put(
        {
            "ops": ops,
            "elapsed": elapsed,
            "samples": dict(rec.samples),
            "counts": dict(rec.counts),
        }
    )


def run_subscriber(db_or_spec, barrier, seconds, results):
    """Follows every test's partition, and measures how late changes arrive."""
    db = make_backend(db_or_spec) if isinstance(db_or_spec, dict) else db_or_spec
    partitions = [test_key(i) for i in range(TESTS)] + ["index"]
    cursors = {p: db.journal_head(p) for p in partitions}
    lags, expired = [], 0
    barrier.wait()
    # Writes made while warming up are stamped before this, and are not counted.
    started = time.time()
    deadline = time.perf_counter() + seconds + 2
    while time.perf_counter() < deadline:
        try:
            changes, cursors = db.journal_read(cursors, timeout=0.5, limit=5000)
        except JournalCursorExpired as err:
            expired += 1
            cursors.update({p: db.journal_head(p) for p in err.partitions})
            continue
        now = time.time()
        for change in changes:
            if change.value is not None and len(change.value) >= 8:
                stamp = struct.unpack(">d", change.value[:8])[0]
                if stamp >= started:
                    lags.append(now - stamp)
    results.put({"lags": lags, "expired": expired})


def percentile(samples, q):
    return (
        statistics.quantiles(samples, n=100, method="inclusive")[q - 1]
        if len(samples) > 1
        else samples[0]
    )


def summarize(samples):
    return {
        "p50_us": round(percentile(samples, 50) * 1e6, 1),
        "p99_us": round(percentile(samples, 99) * 1e6, 1),
    }


def run_one(backend, mode, procs, workload, seconds, lag, rate, args):
    ctx = multiprocessing.get_context("spawn")
    tmp = tempfile.mkdtemp(prefix="hypothesis-bench-")
    run_id = uuid.uuid4().hex[:12]
    spec = {
        "kind": backend,
        "path": os.path.join(tmp, "db.sqlite" if backend == "sqlite" else "examples"),
        "url": args.redis_url if backend == "redis" else args.postgres_url,
        "prefix": f"bench-{run_id}:",
        "namespace": f"bench-{run_id}",
    }
    if backend == "redis":
        from redis import Redis

        Redis.from_url(args.redis_url).flushdb()
    server = None
    backend_db = make_backend(spec)
    if mode == "server":
        server = serve_database(backend_db)
        target = server.client()
    else:
        target = spec
    setup(backend_db, workload)

    barrier = ctx.Barrier(procs + 1 + (1 if lag else 0))
    results = ctx.Queue()
    workers = [
        ctx.Process(
            target=run_worker,
            args=(
                target,
                workload,
                i,
                barrier,
                seconds,
                rate and rate / procs,
                results,
            ),
        )
        for i in range(procs)
    ]
    if lag:
        workers.append(
            ctx.Process(target=run_subscriber, args=(target, barrier, seconds, results))
        )
    for w in workers:
        w.start()
    barrier.wait()
    cpu_start = time.process_time()
    outputs = [results.get(timeout=seconds + 600) for _ in workers]
    server_cpu = time.process_time() - cpu_start
    for w in workers:
        w.join()
    if server is not None:
        server.close()
    shutil.rmtree(tmp, ignore_errors=True)

    worker_outputs = [o for o in outputs if "ops" in o]
    ops = sum(o["ops"] for o in worker_outputs)
    elapsed = max(o["elapsed"] for o in worker_outputs)
    merged = defaultdict(list)
    counts = defaultdict(int)
    for o in worker_outputs:
        for name, samples in o["samples"].items():
            merged[name].extend(samples)
            counts[name] += o["counts"][name]
    row = {
        "backend": backend,
        "mode": mode,
        "procs": procs,
        "workload": workload,
        "rate": rate,
        "seconds": round(elapsed, 2),
        "ops_per_sec": round(ops / elapsed),
        "ops": {
            name: {"count": counts[name], **summarize(s)}
            for name, s in sorted(merged.items())
        },
    }
    if mode == "server":
        row["server_cpu_us_per_op"] = round(server_cpu / max(ops, 1) * 1e6, 1)
    for o in outputs:
        if "lags" in o:
            lags = o["lags"]
            row["lag"] = {
                "delivered": len(lags),
                "expired": o["expired"],
                **(
                    {
                        "p50_ms": round(percentile(lags, 50) * 1e3, 1),
                        "p99_ms": round(percentile(lags, 99) * 1e3, 1),
                        "max_ms": round(max(lags) * 1e3, 1),
                    }
                    if lags
                    else {}
                ),
            }
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", default="memory,directory,sqlite,redis,postgres")
    parser.add_argument("--modes", default="direct,server")
    parser.add_argument("--procs", default="1,4,16,64")
    parser.add_argument("--workloads", default="core,fuzz,startup")
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--lag", action="store_true", help="also measure journal lag")
    parser.add_argument(
        "--rate", type=float, help="total operations per second, for fuzz"
    )
    parser.add_argument("--redis-url", default="redis://localhost:6379")
    parser.add_argument("--postgres-url", default="host=localhost dbname=postgres")
    parser.add_argument("--out", default="results.jsonl")
    args = parser.parse_args()

    for workload in args.workloads.split(","):
        for backend in args.backends.split(","):
            for mode in args.modes.split(","):
                for procs in map(int, args.procs.split(",")):
                    if backend == "memory" and mode == "direct" and procs > 1:
                        continue  # in-memory databases are not shared between processes
                    lag = args.lag and workload == "fuzz" and backend != "directory"
                    rate = args.rate if workload == "fuzz" else None
                    row = run_one(
                        backend, mode, procs, workload, args.seconds, lag, rate, args
                    )
                    with open(args.out, "a") as f:
                        f.write(json.dumps(row) + "\n")
                    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    sys.exit(main())
