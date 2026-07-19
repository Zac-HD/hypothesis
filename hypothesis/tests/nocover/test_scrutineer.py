# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import json
import sys

import pytest

from hypothesis.internal.compat import PYPY, ExceptionGroup
from hypothesis.internal.scrutineer import (
    Tracer,
    get_explaining_locations,
    make_report,
)

from tests.common.utils import skipif_threading

# We skip tracing for explanations under PyPy, where it has a large performance
# impact, or if there is already a trace function (e.g. coverage or a debugger)
pytestmark = pytest.mark.skipif(PYPY or sys.gettrace(), reason="See comment")

BUG_MARKER = "# BUG"
DEADLINE_PRELUDE = """
from datetime import timedelta
from hypothesis.errors import DeadlineExceeded
"""
PRELUDE = """
from hypothesis import Phase, given, settings, strategies as st

@settings(phases=tuple(Phase), derandomize=True)
"""
TRIVIAL = """
@given(st.integers())
def test_reports_branch_in_test(x):
    if x > 10:
        raise AssertionError  # BUG
"""
MULTIPLE_BUGS = """
@given(st.integers(), st.integers())
def test_reports_branch_in_test(x, y):
    if x > 10:
        raise (AssertionError if x % 2 else Exception)  # BUG
"""
FRAGMENTS = (
    pytest.param(TRIVIAL, id="trivial"),
    pytest.param(MULTIPLE_BUGS, id="multiple-bugs"),
)


def get_reports(file_contents, *, testdir):
    # Takes the source code string with "# BUG" comments, and returns a list of
    # multi-line report strings which we expect to see in explain-mode output.
    # The list length is the number of explainable bugs, usually one.
    test_file = str(testdir.makepyfile(file_contents))
    pytest_stdout = str(testdir.runpytest_inprocess(test_file, "--tb=native").stdout)

    # mypyc-compiled black/blib2to3 caches module references at the C level,
    # which can desync from sys.modules after pytester's
    # SysModulesSnapshot.restore() evicts blib2to3.pygram. The next
    # `from . import pygram` then raises KeyError (newer mypyc) or
    # AttributeError (older mypyc).
    upstream_crashes = (
        "AttributeError: module 'blib2to3.pygram' has no attribute 'python_symbols'",
        "KeyError: 'blib2to3.pygram'",
    )
    if any(c in pytest_stdout for c in upstream_crashes):
        pytest.xfail(reason="upstream error in Black/mypyc")

    explanations = {
        i: {(test_file, i)}
        for i, line in enumerate(file_contents.splitlines())
        if line.endswith(BUG_MARKER)
    }
    expected = [
        ("\n".join(r), "\n    | ".join(r))  # single, ExceptionGroup
        for r in make_report(explanations).values()
    ]
    return pytest_stdout, expected


@skipif_threading  # runpytest_inprocess is not thread safe
@pytest.mark.parametrize("code", FRAGMENTS)
def test_explanations(code, testdir):
    pytest_stdout, expected = get_reports(PRELUDE + code, testdir=testdir)
    assert len(expected) == code.count(BUG_MARKER)
    for single, group in expected:
        assert single in pytest_stdout or group in pytest_stdout


@skipif_threading  # runpytest_inprocess is not thread safe
@pytest.mark.parametrize("code", FRAGMENTS)
def test_no_explanations_if_deadline_exceeded(code, testdir):
    code = code.replace("AssertionError", "DeadlineExceeded(timedelta(), timedelta())")
    pytest_stdout, _ = get_reports(DEADLINE_PRELUDE + PRELUDE + code, testdir=testdir)
    assert "Explanation:" not in pytest_stdout


NO_SHOW_CONTEXTLIB = """
from contextlib import contextmanager
from hypothesis import given, strategies as st, Phase, settings

@contextmanager
def ctx():
    yield

@settings(phases=list(Phase))
@given(st.integers())
def test(x):
    with ctx():
        assert x < 100
"""


@skipif_threading  # runpytest_inprocess is not thread safe
@pytest.mark.skipif(PYPY, reason="Tracing is slow under PyPy")
def test_skips_uninformative_locations(testdir):
    pytest_stdout, _ = get_reports(NO_SHOW_CONTEXTLIB, testdir=testdir)
    assert "Explanation:" not in pytest_stdout


NO_SHOW_USER_CLEANUP = """
from hypothesis import Phase, given, settings, strategies as st

class Context:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            cleanup = exc_type.__name__  # only runs after the failure
        return False

@settings(phases=tuple(Phase), derandomize=True)
@given(st.integers())
def test(x):
    with Context():
        assert x < 100
"""


@skipif_threading  # runpytest_inprocess is not thread safe
def test_skips_lines_run_only_after_the_failing_exception(testdir):
    pytest_stdout, _ = get_reports(NO_SHOW_USER_CLEANUP, testdir=testdir)
    assert "Explanation:" not in pytest_stdout


# Two always-failing lines, each reachable from passing-covered code in one of
# the two failing orderings, so that the explanation reports both locations.
DIVERGENCE = """
from hypothesis import Phase, given, settings, strategies as st

def u(fail):
    if fail:
        u_bug = 1  # BUG

def v(fail):
    if fail:
        v_bug = 1  # BUG

@settings(phases=tuple(Phase), derandomize=True)
@given(st.integers())
def test(x):
    fail = x > 10
    if x % 2:
        u(fail)
        v(fail)
    else:
        v(fail)
        u(fail)
    if fail:
        raise AssertionError
"""


@skipif_threading  # runpytest_inprocess is not thread safe
def test_reports_multiple_divergent_locations(testdir):
    test_file = str(testdir.makepyfile(DIVERGENCE))
    pytest_stdout = str(testdir.runpytest_inprocess(test_file, "--tb=native").stdout)
    bug_lines = {
        i for i, line in enumerate(DIVERGENCE.splitlines()) if line.endswith(BUG_MARKER)
    }
    assert all(f"{test_file}:{lineno}" in pytest_stdout for lineno in bug_lines)


class ImmutableError(Exception):
    def __setattr__(self, name, value):
        raise AttributeError(name)


@pytest.mark.parametrize("exc_type", [ValueError, ImmutableError])
def test_tracer_excludes_branches_first_run_after_exception(exc_type):
    tracer = Tracer(should_trace=False)
    code = compile("pass", "fake_file.py", "exec")
    tracer.trace_line(code, 1)
    tracer.trace_line(code, 2)
    try:
        raise exc_type()
    except exc_type as e:
        exc = e
    tracer.trace_raise(code, 0, exc)
    tracer.trace_line(code, 3)

    pre_raise = {
        (None, ("fake_file.py", 1)),
        (("fake_file.py", 1), ("fake_file.py", 2)),
    }
    assert tracer.branches == pre_raise | {(("fake_file.py", 2), ("fake_file.py", 3))}
    assert tracer.branches_before(exc) == pre_raise
    assert tracer.branches_before(ExceptionGroup("wrapped", [exc])) == pre_raise
    # an exception we never saw raised leaves the trace unfiltered
    assert tracer.branches_before(KeyError()) == tracer.branches
    # as does an exception raised under a previous Tracer
    other = Tracer(should_trace=False)
    other.trace_line(code, 1)
    assert other.branches_before(exc) == other.branches


def test_tracer_ranks_locations_by_first_execution():
    tracer = Tracer(should_trace=False)
    code = compile("pass", "fake_file.py", "exec")
    for lineno in (3, 1, 2, 1):
        tracer.trace_line(code, lineno)
    assert tracer.location_ranks() == {
        ("fake_file.py", 3): 0,
        ("fake_file.py", 1): 1,
        ("fake_file.py", 2): 2,
    }


def test_report_truncates_long_reports():
    explanations = {"origin": [(__file__, n) for n in range(1, 15)]}
    report_lines = [line.strip() for line in make_report(explanations)["origin"][2:]]
    assert report_lines == [f"{__file__}:{n}" for n in range(1, 11)] + [
        "[ ... 4 lines omitted; use settings.verbosity=verbose to show ]"
    ]
    # eleven lines fit without truncation
    explanations = {"origin": [(__file__, n) for n in range(1, 12)]}
    assert len(make_report(explanations)["origin"][2:]) == 11


def test_empty_traces_are_dropped():
    # e.g. a failing example where every line ran only after the exception
    assert get_explaining_locations({None: set(), "origin": {frozenset()}}) == {}


def test_report_orders_lines_by_first_execution():
    a, b, c = (__file__, 10), (__file__, 20), (__file__, 30)
    ranks = {"origin": {c: 0, a: 1}}
    report = make_report({"origin": [a, b, c]}, location_ranks=ranks)
    lines = [line.strip() for line in report["origin"][2:]]
    # c ran first, then a; b wasn't seen executing so it sorts last
    assert lines == [f"{__file__}:30", f"{__file__}:10", f"{__file__}:20"]


def test_report_truncation_prefers_dropping_stdlib_lines():
    local = [(__file__, n) for n in range(4)]
    site = [(pytest.__file__, n) for n in range(4)]
    stdlib = [(json.__file__, n) for n in range(6)]
    lines = [
        line.strip()
        for line in make_report({"origin": local + site + stdlib})["origin"][2:]
    ]
    # all local and site-packages lines fit, plus the two earliest stdlib lines
    assert (
        lines[-1] == "[ ... 4 lines omitted; use settings.verbosity=verbose to show ]"
    )
    kept = lines[:-1]
    assert len(kept) == 10
    for fname, lineno in local + site + stdlib[:2]:
        assert f"{fname}:{lineno}" in kept


def test_report_truncation_never_drops_local_lines():
    local = [(__file__, n) for n in range(2)]
    site = [(pytest.__file__, n) for n in range(30)]
    lines = [
        line.strip() for line in make_report({"origin": local + site})["origin"][2:]
    ]
    # both local lines survive, along with the eight earliest site-packages lines
    assert (
        lines[-1] == "[ ... 22 lines omitted; use settings.verbosity=verbose to show ]"
    )
    kept = lines[:-1]
    assert len(kept) == 10
    for fname, lineno in local + site[:8]:
        assert f"{fname}:{lineno}" in kept
