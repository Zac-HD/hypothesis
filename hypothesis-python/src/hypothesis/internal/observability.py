# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Observability tools to spit out analysis-ready tables, one row per test case."""

import json
import os
import sys
import warnings
from datetime import date, timedelta
from typing import Callable, Dict, List, Optional

from hypothesis.configuration import storage_directory
from hypothesis.internal.conjecture.data import ConjectureData, Status

TESTCASE_CALLBACKS: List[Callable[[dict], None]] = []


def deliver_json_blob(value: dict) -> None:
    for callback in TESTCASE_CALLBACKS:
        callback(value)


def make_testcase(
    *,
    start_timestamp: float,
    test_name_or_nodeid: str,
    data: ConjectureData,
    how_generated: str = "unknown",
    string_repr: str = "<unknown>",
    arguments: Optional[dict] = None,
    metadata: Optional[dict] = None,
    coverage: Optional[Dict[str, List[int]]] = None,
) -> dict:
    if data.interesting_origin:
        status_reason = str(data.interesting_origin)
    else:
        status_reason = str(data.events.pop("invalid because", ""))

    return {
        "type": "test_case",
        "run_start": start_timestamp,
        "property": test_name_or_nodeid,
        "status": {
            Status.OVERRUN: "gave_up",
            Status.INVALID: "gave_up",
            Status.VALID: "passed",
            Status.INTERESTING: "failed",
        }[data.status],
        "status_reason": status_reason,
        "representation": string_repr,
        "arguments": arguments or {},
        "how_generated": how_generated,  # iid, mutation, etc.
        "features": {
            **{
                f"target:{k}".strip(":"): v for k, v in data.target_observations.items()
            },
            **data.events,
        },
        "metadata": {
            **(metadata or {}),
            "traceback": getattr(data.extra_information, "_expected_traceback", None),
        },
        "coverage": coverage,
    }


_WROTE_TO = set()


def _deliver_to_file(value):  # pragma: no cover
    kind = "testcases" if value["type"] == "test_case" else "info"
    fname = storage_directory("observed", f"{date.today().isoformat()}_{kind}.jsonl")
    fname.parent.mkdir(exist_ok=True)
    _WROTE_TO.add(fname)
    with fname.open(mode="a") as f:
        f.write(json.dumps(value) + "\n")


if "HYPOTHESIS_EXPERIMENTAL_OBSERVABILITY" in os.environ:  # pragma: no cover
    TESTCASE_CALLBACKS.append(_deliver_to_file)

    # Remove files more than a week old, to cap the size on disk
    max_age = (date.today() - timedelta(days=8)).isoformat()
    for f in storage_directory("observed").glob("*.jsonl"):
        if f.stem < max_age:  # pragma: no branch
            f.unlink(missing_ok=True)


def maybe_recommend_tyche():
    # Introspect whether we're being run from a VS Code testing session; if so we'll
    # recommend installing the Tyche frontend.  For module names, see
    # https://github.com/microsoft/vscode-python/tree/main/pythonFiles
    # NOTE: we'd be delighted to support other editors such as PyCharm too :-)
    if "testing_tools" in sys.modules and {
        "vscode_pytest",
        "unittestadapter",
    }.intersection(sys.modules):
        try:
            # FIXME: this would have the side-effect of loading the shim, when we really
            #        only want to check whether it's already present.  Let's wait and
            #        see if Tyche can integrate everything into the VSCode plugin, as
            #        for microsoft's Python plugin which we detected above.
            #
            # pip install git+https://github.com/tyche-pbt/tyche-hypothesis.git
            pass
        except ImportError:
            # TODO: this whole setup assumes that:
            #   (a) we do in fact endorse Tyche for ~all users in VS Code,
            #       which I'm unsure of - does it get out of the way enough
            #       when it's not providing much value?
            #       We need some easy way to disable this warning if not!
            #   (b) installing the VSC plugin is sufficient, i.e. that hooks
            #       up an entrypoint which will add a delivery callback to
            #       TESTCASE_CALLBACKS.
            warnings.warn(
                "We recommend using the HarrisonGoldstein.tyche extension when running "
                "property-based tests in VS Code; it provides some nice visualizations "
                "including highlighting lines of code covered by passing and/or failing "
                "inputs to your test function.",
                category=UserWarning,
                stacklevel=4,
            )
