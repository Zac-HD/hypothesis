"""
An interactive tutorial for Hypothesis, inspired by Examplar.
https://blog.brownplt.org/2024/01/01/examplar.html

1. Introduction, scene-setting text about sorting.
2. Fill in the test body, with the known-good strategy and good implementation
3. Just Examplar: but when you write inputs, we reveal strategies (based on hand-crafted predicates)
   demo: https://claude.site/artifacts/baace119-841a-4bf4-a47b-258351d96b91
4. PropXamplar: given a good strategy, write test body to distinguish good from bad sorting functions

Let's put this at the end of the introductory tutorial!


-------------

We might eventually develop additional tutorials leaning on other strategies
or other kinds of properties, maybe a slightly tricky encoder/decoder pair?

There's also the 'strategies flashcards / fill-in-the-blank' minigame, which
might be nice as a revision option for some people but isn't suitable as an
introductory tutorial (too much assumed knowledge).

=======================


Plan: let's write some tests for sorting.
(chosen _because_ it's so well-known; you can focus on Hypothesis rather than sorting)

* get the very basics working first
    * strategies
        - integers
        - mixed integers and floats (must exclude nan)
        - list[int|float] | list[str] - not just numeric, testing with lists that can't all be sorted across runs

    * test function for numeric sorting only; chaffs include
        - return []
        - return input
        - return sorted(set(input))
        - that, plus max-suffix; that plus min-prefix

    * [extension] write evil implementations which pass incomplete tests
        - checks ordered only
        - checks same set of elems only
        - checks same length only
        - checks ordered, length, set

* then extensions
    * something with key functions?
    * another level: mean and median with floats (hard!)

=======================

Claude helped me make a "fill in the blank" puzzle:
https://claude.site/artifacts/ff29142d-6544-4a6f-b2ab-3468b4188cba

I don't think this is a good _introduction_ to strategies, but might
be nice as an optional "come back and check what you've learned" option
for people who like quizzes and flashcards etc.  Maybe we should have a
toggle for fill-blanks vs flashcard style?  Click-to-reveal would do it.

=> this should go at end of the "builtin strategies" mini-tutorial.
"""

from collections import Counter
from typing import Callable, Collection, TypeVar

from hypothesis import Phase, given, settings, strategies as st
from hypothesis.core import failure_exceptions_to_catch

#################################################


T = TypeVar("T")


def sort_really_lazy(ls: Collection[T], /) -> list[T]:
    return []


def sort_lazy(ls: Collection[T], /) -> list[T]:
    return ls


def sort_unique(ls: Collection[T], /) -> list[T]:
    return sorted(set(ls))


def sort_first_elem(ls: Collection[T], /) -> list[T]:
    return [ls[0] for _ in ls]


def sort_unique_and_extend(ls: Collection[T], /) -> list[T]:
    return sorted(set(ls)) + [max(ls)] * (len(ls) - len(set(ls)))


sorting_impls = (
    (sorted, True),  # tests should pass for this one
    (sort_really_lazy, False),  # ...but not for the others
    (sort_lazy, False),
    (sort_unique, False),
    (sort_first_elem, False),
    (sort_unique_and_extend, False),
)


#################################################


def assert_ordered(arg: Collection[T], out: list[T]) -> None:
    assert all(a <= b for a, b in zip(out, out[1:]))


def assert_same_len(arg: Collection[T], out: list[T]) -> None:
    assert len(arg) == len(out)


def assert_same_set(arg: Collection[T], out: list[T]) -> None:
    assert set(arg) == set(out)


def assert_ordered_and_same_len(arg: Collection[T], out: list[T]) -> None:
    assert len(arg) == len(out)
    assert all(a <= b for a, b in zip(out, out[1:]))


def assert_ordered_and_same_set(arg: Collection[T], out: list[T]) -> None:
    assert set(arg) == set(out)
    assert all(a <= b for a, b in zip(out, out[1:]))


def assert_ordered_and_same_len_and_set(arg: Collection[T], out: list[T]) -> None:
    assert len(arg) == len(out)
    assert set(arg) == set(out)
    assert all(a <= b for a, b in zip(out, out[1:]))


def assert_correct_sort(arg: Collection[T], out: list[T]) -> None:
    assert Counter(arg) == Counter(out)
    assert all(a <= b for a, b in zip(out, out[1:]))


sorting_check_fns = [
    (assert_ordered, False),  # returns_empty passes
    (assert_same_len, False),  # returns_arg passes
    (assert_ordered_and_same_len, False),  # duplicate_first
    (assert_ordered_and_same_set, False),  # sorts_set_only
    (assert_ordered_and_same_len_and_set, False),  # dupes
    (assert_correct_sort, True),  # correct check
]


#################################################


sorting_strats = [
    st.builds(list),  # empty list only
    st.lists(st.integers(), max_size=1), # can't be unsorted
    st.lists(st.integers()).map(sorted),  # already sorted
    st.lists(st.integers(), unique=True),  # no duplicates
    st.builds((lambda x, n: [x for _ in range(n)]),
              x=st.integers(), n=st.integers(0, 5)),  # same value at every index
    # st.lists(st.integers()),  # the good stuff
]


#################################################


T_arg = TypeVar("T_arg")
T_out = TypeVar("T_out")


def runner(
    strategy: st.SearchStrategy[T_arg],
    impl_fn: Callable[[T_arg], T_out],
    assert_fn: Callable[[T_arg, T_out], None],
) -> bool:
    """Execute a PBT for this combination."""
    # print(strategy, impl_fn.__name__, assert_fn.__name__)

    @given(strategy)
    @settings(derandomize=True, phases=[Phase.generate])
    def wrapped(arg):
        out = impl_fn(arg)
        assert assert_fn(arg, out) is None

    try:
        wrapped()
    except failure_exceptions_to_catch():
        return False
    return True


#################################################


def selftest():
    # Varying strategies: the correct test passes with any strategy
    for strat in sorting_strats:
        assert runner(strat, sorted, assert_correct_sort)

    # Varying impls: reference strategy + assert is a correct classifier
    for impl_fn, should_pass in sorting_impls:
        got = runner(st.lists(st.integers()), impl_fn, assert_correct_sort)
        assert got is should_pass


    # Varying asserts: each incorrect assert leads to at least one missed alarm
    assert_results = [
        tuple(runner(st.lists(st.integers()), i, a) for i, _ in sorting_impls)
        for a, ok in sorting_check_fns if not ok
    ]
    assert len(set(assert_results)) == len(assert_results)  # unique signatures
    assert all(False in row for row in assert_results)

    # Varying strategies: each incorrect strategy leads to at least one missed alarm
    strat_results = [
        tuple(runner(s, i, assert_correct_sort) for i, ok in sorting_impls if not ok)
        for s in sorting_strats
    ]
    assert len(set(strat_results)) == len(strat_results)  # unique signatures
    assert all(set(row) == {True, False} for row in strat_results)  # and some passes!


if __name__ == "__main__":
    selftest()
