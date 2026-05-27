# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import asyncio
from collections.abc import AsyncIterator, Iterator
from inspect import (
    isasyncgenfunction,
    iscoroutinefunction,
    isgeneratorfunction,
    signature,
)

import pytest

from hypothesis import Verbosity, assume, find, given, settings, strategies as st
from hypothesis.errors import InvalidArgument, InvalidState
from hypothesis.reporting import with_reporter
from hypothesis.strategies import booleans, functions, integers

from tests.common.debug import check_can_generate_examples


def func_a():
    pass


@given(functions(like=func_a, returns=booleans()))
def test_functions_no_args(f):
    assert f.__name__ == "func_a"
    assert f is not func_a
    assert isinstance(f(), bool)


def func_b(a, b, c):
    pass


@given(functions(like=func_b, returns=booleans()))
def test_functions_with_args(f):
    assert f.__name__ == "func_b"
    assert f is not func_b
    with pytest.raises(TypeError):
        f()
    assert isinstance(f(1, 2, 3), bool)


def func_c(**kwargs):
    pass


@given(functions(like=func_c, returns=booleans()))
def test_functions_kw_args(f):
    assert f.__name__ == "func_c"
    assert f is not func_c
    with pytest.raises(TypeError):
        f(1, 2, 3)
    assert isinstance(f(a=1, b=2, c=3), bool)


@given(functions(like=lambda: None, returns=booleans()))
def test_functions_argless_lambda(f):
    assert f.__name__ == "<lambda>"
    with pytest.raises(TypeError):
        f(1)
    assert isinstance(f(), bool)


@given(functions(like=lambda a: None, returns=booleans()))
def test_functions_lambda_with_arg(f):
    assert f.__name__ == "<lambda>"
    with pytest.raises(TypeError):
        f()
    assert isinstance(f(1), bool)


@pytest.mark.parametrize(
    "like,returns,pure",
    [
        (None, booleans(), False),
        (lambda: None, "not a strategy", True),
        (lambda: None, booleans(), None),
    ],
)
def test_invalid_arguments(like, returns, pure):
    with pytest.raises(InvalidArgument):
        check_can_generate_examples(functions(like=like, returns=returns, pure=pure))


def func_returns_str() -> str:
    return "a string"


@given(functions(like=func_returns_str))
def test_functions_strategy_return_type_inference(f):
    result = f()
    assume(result != "a string")
    assert isinstance(result, str)


def test_functions_valid_within_given_invalid_outside():
    cache = None

    @given(functions())
    def t(f):
        nonlocal cache
        cache = f
        assert f() is None

    t()
    with pytest.raises(InvalidState):
        cache()


def test_can_call_default_like_arg():
    # This test is somewhat silly, but coverage complains about the uncovered
    # branch for calling it otherwise and alternative workarounds are worse.
    like, returns, pure = signature(functions).parameters.values()
    assert like.default() is None
    assert returns.default is ...
    assert pure.default is False


def func(arg, *, kwonly_arg):
    pass


@given(functions(like=func))
def test_functions_strategy_with_kwonly_args(f):
    with pytest.raises(TypeError):
        f(1, 2)
    f(1, kwonly_arg=2)
    f(kwonly_arg=2, arg=1)


def pure_func(arg1, arg2):
    pass


@given(
    f=functions(like=pure_func, returns=integers(), pure=True),
    arg1=integers(),
    arg2=integers(),
)
def test_functions_pure_with_same_args(f, arg1, arg2):
    # Same regardless of calling convention, unlike functools.lru_cache()
    expected = f(arg1, arg2)
    assert f(arg1, arg2) == expected
    assert f(arg1, arg2=arg2) == expected
    assert f(arg1=arg1, arg2=arg2) == expected
    assert f(arg2=arg2, arg1=arg1) == expected


@given(
    f=functions(like=pure_func, returns=integers(), pure=True),
    arg1=integers(),
    arg2=integers(),
)
def test_functions_pure_with_different_args(f, arg1, arg2):
    r1 = f(arg1, arg2)
    r2 = f(arg2, arg1)
    assume(r1 != r2)
    # If this is never true, the test will fail with Unsatisfiable


@given(
    f1=functions(like=pure_func, returns=integers(), pure=True),
    f2=functions(like=pure_func, returns=integers(), pure=True),
)
def test_functions_pure_two_functions_different_args_different_result(f1, f2):
    r1 = f1(1, 2)
    r2 = f2(3, 4)
    assume(r1 != r2)
    # If this is never true, the test will fail with Unsatisfiable


@given(
    f1=functions(like=pure_func, returns=integers(), pure=True),
    f2=functions(like=pure_func, returns=integers(), pure=True),
    arg1=integers(),
    arg2=integers(),
)
def test_functions_pure_two_functions_same_args_different_result(f1, f2, arg1, arg2):
    r1 = f1(arg1, arg2)
    r2 = f2(arg1, arg2)
    assume(r1 != r2)
    # If this is never true, the test will fail with Unsatisfiable


@settings(verbosity=Verbosity.verbose)
@given(functions(pure=False))
def test_functions_note_all_calls_to_impure_functions(f):
    ls = []
    with with_reporter(ls.append):
        f()
        f()
    assert len(ls) == 2


@settings(verbosity=Verbosity.verbose)
@given(functions(pure=True))
def test_functions_note_only_first_to_pure_functions(f):
    ls = []
    with with_reporter(ls.append):
        f()
        f()
    assert len(ls) == 1


def test_functions_supports_find():
    f = find(
        st.functions(like=pure_func, returns=st.integers(), pure=True), lambda x: True
    )
    with pytest.raises(InvalidState):
        f(1, 2)
    assert f.__name__ == pure_func.__name__


async def async_func_a(a, b) -> int: ...


@given(functions(like=async_func_a))
def test_functions_async(f):
    assert iscoroutinefunction(f)
    assert f.__name__ == "async_func_a"
    assert list(signature(f).parameters) == ["a", "b"]
    with pytest.raises(TypeError):
        f(1)
    result = asyncio.run(f(1, 2))
    assert isinstance(result, int)


@given(functions(like=async_func_a, returns=booleans()))
def test_functions_async_explicit_returns(f):
    assert isinstance(asyncio.run(f(1, 2)), bool)


@given(f=functions(like=async_func_a, returns=integers(), pure=True))
def test_functions_async_pure(f):
    assert asyncio.run(f(1, 2)) == asyncio.run(f(1, 2))


def gen_func(a) -> Iterator[int]:
    yield a


@given(functions(like=gen_func))
def test_functions_generator(f):
    assert isgeneratorfunction(f)
    assert f.__name__ == "gen_func"
    assert list(signature(f).parameters) == ["a"]
    with pytest.raises(TypeError):
        list(f())
    result = list(f(1))
    assert isinstance(result, list)
    assert all(isinstance(x, int) for x in result)


@given(functions(like=gen_func, returns=booleans()))
def test_functions_generator_explicit_returns(f):
    assert all(isinstance(x, bool) for x in f(1))


def gen_no_annotation(a):
    yield a


@given(functions(like=gen_no_annotation))
def test_functions_generator_infers_none_without_yield_type(f):
    assert isgeneratorfunction(f)
    assert all(x is None for x in f(1))


@given(f=functions(like=gen_func, returns=integers(), pure=True))
def test_functions_generator_pure(f):
    assert list(f(1)) == list(f(1))


async def agen_func(a) -> AsyncIterator[int]:
    yield a


@given(functions(like=agen_func))
def test_functions_async_generator(f):
    assert isasyncgenfunction(f)
    assert f.__name__ == "agen_func"

    async def collect():
        return [x async for x in f(1)]

    result = asyncio.run(collect())
    assert all(isinstance(x, int) for x in result)


def test_functions_async_valid_within_given_invalid_outside():
    cache = None

    @given(functions(like=async_func_a, returns=integers()))
    def t(f):
        nonlocal cache
        cache = f
        assert isinstance(asyncio.run(f(1, 2)), int)

    t()
    with pytest.raises(InvalidState):
        asyncio.run(cache(1, 2))
