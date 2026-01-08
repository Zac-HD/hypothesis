# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import pytest

from hypothesis import given, strategies as st
from hypothesis.errors import InvalidArgument

from tests.common.debug import check_can_generate_examples, find_any, minimal


@given(st.recursive(st.booleans(), st.lists, max_leaves=10))
def test_respects_leaf_limit(xs):
    def flatten(x):
        if isinstance(x, list):
            return sum(map(flatten, x), [])
        else:
            return [x]

    assert len(flatten(xs)) <= 10


def test_can_find_nested():
    x = minimal(
        st.recursive(st.booleans(), lambda x: st.tuples(x, x)),
        lambda x: isinstance(x, tuple) and isinstance(x[0], tuple),
    )

    assert x == ((False, False), False)


def test_recursive_call_validates_expand_returns_strategies():
    with pytest.raises(InvalidArgument):
        check_can_generate_examples(st.recursive(st.booleans(), lambda x: 1))


def test_recursive_call_validates_base_is_strategy():
    x = st.recursive(1, lambda x: st.none())
    with pytest.raises(InvalidArgument):
        check_can_generate_examples(x)


def test_can_find_exactly_max_leaves():
    strat = st.recursive(st.none(), lambda x: st.tuples(x, x), max_leaves=5)

    def enough_leaves(t):
        print(t)
        count = 0
        stack = [t]
        while stack:
            s = stack.pop()
            if s is None:
                count += 1
            else:
                stack.extend(s)
        return count >= 5

    find_any(strat, enough_leaves)


@given(st.recursive(st.none(), lambda x: st.tuples(x, x), max_leaves=1))
def test_can_exclude_branching_with_max_leaves(t):
    assert t is None


@given(st.recursive(st.none(), lambda x: st.one_of(x, x)))
def test_issue_1502_regression(s):
    pass


def test_recursive_with_max_leaves_can_generate_nested_structures():
    # Regression test for issue #4638
    # st.recursive() with max_leaves should be able to generate nested structures
    useful_values = st.booleans() | st.integers() | st.text()
    useful_list = st.lists(st.deferred(lambda: useful_values), max_size=5)
    useful_dict = st.dictionaries(st.text(), st.deferred(lambda: useful_values), max_size=5)

    json = st.dictionaries(
        st.text(),
        st.recursive(
            useful_values,
            lambda x: useful_list | useful_dict,
            max_leaves=20,
        ),
        max_size=10,
    )

    # Should be able to find examples with nested lists
    def has_list_value(d):
        return any(isinstance(v, list) for v in d.values())

    find_any(json, has_list_value)


@pytest.mark.parametrize(
    "s",
    [
        st.recursive(None, st.lists),
        st.recursive(st.none(), lambda x: None),
        st.recursive(st.none(), st.lists, max_leaves=-1),
        st.recursive(st.none(), st.lists, max_leaves=0),
        st.recursive(st.none(), st.lists, max_leaves=1.0),
    ],
)
def test_invalid_args(s):
    with pytest.raises(InvalidArgument):
        check_can_generate_examples(s)
