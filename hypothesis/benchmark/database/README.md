This directory benchmarks the database backends, for the design in
`guides/database-design.md`. Results are in section 12 of that document.

To run everything:

```
pip install redis "psycopg[binary]"
python bench.py --redis-url redis://localhost:6379 --postgres-url "host=localhost dbname=postgres"
```

Each configuration appends one JSON line to `results.jsonl`. Use `--backends`,
`--modes`, `--procs`, `--workloads`, and `--seconds` to run a subset, and
`python report.py results.jsonl` to print tables.

Workloads:

- `core`: what plain Hypothesis does. Read a test's three keys with one
  `read_many`, or with three `fetch` calls every tenth time. Save a failure 5%
  of the time.
- `fuzz`: what a HypoFuzz worker does, as fast as possible: an observation
  append, a progress put, and sometimes a corpus put, an index put, or a report
  append. Each worker uses 10 of 100 tests.
- `startup`: load a corpus of 200 entries with 3 kB observations. Compare one
  `map_items` with the old layout, which needs one `fetch` per entry.

Modes:

- `direct`: each process opens the backend itself.
- `server`: the parent process serves the backend, and workers use a
  `RemoteDatabase`. Queued writes return at once, so the throughput figures
  include the final `flush`.

`--lag` adds a process that follows every test's journal, and reports how long
each change took to arrive. Pair it with `--rate`, which paces the `fuzz`
workload: at full speed, the lag mostly measures the write queue.

Backends run on the same machine as the workers, so the figures include
contention for CPU. `server_cpu_us_per_op` is the CPU time of the serving
process, which does not include the database server's own CPU time.
