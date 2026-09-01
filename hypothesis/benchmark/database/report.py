# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Prints Markdown tables from the results that bench.py writes."""

import json
import sys
from collections import defaultdict

BACKENDS = ["memory", "directory", "sqlite", "redis", "postgres"]


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def order(row):
    return (BACKENDS.index(row["backend"]), row["mode"])


def main(path, latency_procs=4):
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    throughput = [r for r in rows if not r.get("rate")]
    for workload in ["core", "fuzz", "startup"]:
        selected = [r for r in throughput if r["workload"] == workload]
        if not selected:
            continue
        procs = sorted({r["procs"] for r in selected})
        by_config = defaultdict(dict)
        for r in selected:
            by_config[(r["backend"], r["mode"])][r["procs"]] = r
        print(f"\n### {workload}: operations per second\n")
        body = []
        for (backend, mode), runs in sorted(
            by_config.items(),
            key=lambda kv: order({"backend": kv[0][0], "mode": kv[0][1]}),
        ):
            cells = [f"{runs[p]['ops_per_sec']:,}" if p in runs else "" for p in procs]
            cpu = [runs[p].get("server_cpu_us_per_op") for p in procs if p in runs]
            body.append(
                [backend, mode, *cells, cpu[-1] if cpu and cpu[-1] is not None else ""]
            )
        print(
            table(
                ["backend", "mode", *[f"{p} procs" for p in procs], "server µs/op"],
                body,
            )
        )

        print(
            f"\n### {workload}: latency in µs at {latency_procs} processes, p50 / p99\n"
        )
        names = sorted({n for r in selected for n in r["ops"]})
        body = []
        for r in sorted(selected, key=order):
            if r["procs"] != latency_procs:
                continue
            cells = []
            for name in names:
                s = r["ops"].get(name)
                cells.append(f"{s['p50_us']:,.0f} / {s['p99_us']:,.0f}" if s else "")
            body.append([r["backend"], r["mode"], *cells])
        if body:
            print(
                table(["backend", "mode", *[n.split(": ", 1)[1] for n in names]], body)
            )

    lagged = [r for r in rows if "lag" in r]
    if lagged:
        print("\n### journal lag, in milliseconds\n")
        body = [
            [
                r["backend"],
                r["mode"],
                r["procs"],
                f"{r['rate']:,.0f}" if r.get("rate") else "max",
                f"{r['ops_per_sec']:,}",
                r["lag"]["delivered"],
                r["lag"].get("p50_ms", ""),
                r["lag"].get("p99_ms", ""),
                r["lag"].get("max_ms", ""),
            ]
            for r in sorted(lagged, key=lambda r: (order(r), r["procs"]))
        ]
        print(
            table(
                [
                    "backend",
                    "mode",
                    "procs",
                    "target ops/s",
                    "ops/s",
                    "changes",
                    "p50",
                    "p99",
                    "max",
                ],
                body,
            )
        )


if __name__ == "__main__":
    main(*sys.argv[1:2])
