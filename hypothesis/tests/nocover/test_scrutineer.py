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
import sysconfig

import pytest

from hypothesis import given, note, settings, strategies as st
from hypothesis.internal.compat import PYPY, ExceptionGroup
from hypothesis.internal.scrutineer import Tracer, make_report
from hypothesis.vendor import pretty

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
def test_annotates_earliest_divergence(testdir):
    test_file = str(testdir.makepyfile(DIVERGENCE))
    pytest_stdout = str(testdir.runpytest_inprocess(test_file, "--tb=native").stdout)
    bug_lines = {
        i for i, line in enumerate(DIVERGENCE.splitlines()) if line.endswith(BUG_MARKER)
    }
    assert all(f"{test_file}:{lineno}" in pytest_stdout for lineno in bug_lines)
    annotated = [
        line
        for line in pytest_stdout.splitlines()
        if line.endswith("(earliest divergence)")
    ]
    assert len(annotated) == 1
    assert any(f"{test_file}:{lineno} " in annotated[0] for lineno in bug_lines)


def test_tracer_excludes_branches_first_run_after_exception():
    tracer = Tracer(should_trace=False)
    code = compile("pass", "fake_file.py", "exec")
    tracer.trace_line(code, 1)
    tracer.trace_line(code, 2)
    exc = ValueError()
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


def test_report_shows_ten_lines_before_truncating():
    explanations = {"origin": [(__file__, n) for n in range(1, 15)]}
    report_lines = [line.strip() for line in make_report(explanations)["origin"][2:]]
    assert report_lines[:10] == [f"{__file__}:{n}" for n in range(1, 11)]
    assert report_lines[10:] == ["(and 4 more with settings.verbosity >= verbose)"]


def test_report_annotates_earliest_divergence():
    a, b = (__file__, 10), (__file__, 20)
    ranks = {"origin": {b: 0, a: 1}}
    report = make_report({"origin": [a, b]}, location_ranks=ranks)
    lines = [line.strip() for line in report["origin"][2:]]
    assert lines == [f"{__file__}:10", f"{__file__}:20  (earliest divergence)"]

    # single-location reports are not annotated
    report = make_report({"origin": [b]}, location_ranks=ranks)
    assert report["origin"][2].strip() == f"{__file__}:20"


@given(st.randoms())
@settings(max_examples=5)
def test_report_sort(random):
    # show local files first, then site-packages, then stdlib

    lines = [
        # local
        (__file__, 10),
        # site-packages
        (pytest.__file__, 123),
        (pytest.__file__, 124),
        # stdlib
        (json.__file__, 43),
        (json.__file__, 42),
    ]
    random.shuffle(lines)
    explanations = {"origin": lines}
    report = make_report(explanations)
    report_lines = report["origin"][2:]
    report_lines = [line.strip() for line in report_lines]

    expected_lines = [
        f"{__file__}:10",
        f"{pytest.__file__}:123",
        f"{pytest.__file__}:124",
        f"{json.__file__}:42",
        f"{json.__file__}:43",
    ]

    note(f"sysconfig.get_paths(): {pretty.pretty(sysconfig.get_paths())}")
    note(f"actual lines: {pretty.pretty(report_lines)}")
    note(f"expected lines: {pretty.pretty(expected_lines)}")

    assert report_lines == expected_lines
