# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from inspect import signature

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
@given(functions(pure=True, returns=integers()))
def test_functions_are_summarised_after_the_test_not_noted_per_call(f):
    # We describe the whole function once the test has finished, instead of
    # cluttering the output with a note on every call.
    ls = []
    with with_reporter(ls.append):
        f()
        f()
    assert ls == []


def _falsifying_output(strategy, body):
    # Run a failing test, and return the reported failing example together with
    # the description of the generated function.
    @settings(derandomize=True)
    @given(f=strategy)
    def test(f):
        body(f)

    with pytest.raises(AssertionError) as excinfo:
        test()
    return "\n".join(getattr(excinfo.value, "__notes__", []))


def fails(body):
    # Wrap a body which returns True for the cases we want to see reported.
    def inner(f):
        assert not body(f)

    return inner


def always_fails(call):
    # Wrap a body which should be reported however the calls turn out.
    def inner(f):
        call(f)
        raise AssertionError

    return inner


def func_one_arg(x):
    pass


def test_pure_function_shown_as_dict_lookup():
    out = _falsifying_output(
        functions(like=lambda x: None, returns=integers(0, 9), pure=True),
        fails(lambda f: f(1) != f(2)),
    )
    assert "lambda x: {1: 0, 2: 1}[x]" in out


def test_pure_function_uses_tuple_key_for_several_args():
    out = _falsifying_output(
        functions(like=lambda x, y: None, returns=integers(0, 9), pure=True),
        fails(lambda f: f(1, 2) != f(3, 4)),
    )
    assert "lambda x, y: {(1, 2): 0, (3, 4): 1}[(x, y)]" in out


def test_pure_function_with_kwonly_args():
    out = _falsifying_output(
        functions(like=func, returns=integers(0, 9), pure=True),
        fails(lambda f: f(1, kwonly_arg=2) != f(3, kwonly_arg=4)),
    )
    assert "lambda arg, *, kwonly_arg: {(1, 2): 0, (3, 4): 1}[(arg, kwonly_arg)]" in out


def func_posonly(a, b, /, c):
    pass


def func_args_kwonly(*args, k):
    pass


def test_pure_function_with_positional_only_args():
    out = _falsifying_output(
        functions(like=func_posonly, returns=integers(0, 9), pure=True),
        fails(lambda f: f(1, 2, 3) != f(4, 5, 6)),
    )
    assert "lambda a, b, /, c: {(1, 2, 3): 0, (4, 5, 6): 1}[(a, b, c)]" in out


def test_pure_function_with_var_positional_args():
    out = _falsifying_output(
        functions(like=lambda *args: None, returns=integers(0, 9), pure=True),
        fails(lambda f: f(1, 2) != f(3)),
    )
    assert "lambda *args: {(1, 2): 0, (3,): 1}[args]" in out


def test_pure_function_with_var_positional_and_kwonly_args():
    out = _falsifying_output(
        functions(like=func_args_kwonly, returns=integers(0, 9), pure=True),
        fails(lambda f: f(1, 2, k=3) != f(4, k=5)),
    )
    assert "lambda *args, k: {((1, 2), 3): 0, ((4,), 5): 1}[(args, k)]" in out


def test_pure_function_key_applies_defaults():
    # Passing a default explicitly is the same call, so it shares a cache entry
    # - and therefore appears only once in the mapping.
    out = _falsifying_output(
        functions(like=lambda a, b=5: None, returns=integers(0, 9), pure=True),
        fails(lambda f: f(1) == f(1, 5) != f(1, 6)),
    )
    assert "lambda a, b: {(1, 5): 0, (1, 6): 1}[(a, b)]" in out


def test_pure_function_with_kwargs_falls_back_to_call_log():
    # The **kwargs dict is unhashable, so a lookup expression would have to
    # flatten it to sorted pairs - too ugly to be worth showing.
    out = _falsifying_output(
        functions(like=func_c, returns=integers(0, 9), pure=True),
        fails(lambda f: f(a=1) != f(a=2)),
    )
    assert "lambda" not in out
    assert "Called function: func_c(a=1) -> 0" in out
    assert "Called function: func_c(a=2) -> 1" in out


def test_pure_function_with_kwargs_still_shown_as_constant():
    out = _falsifying_output(
        functions(like=func_c, returns=integers(0, 9), pure=True),
        fails(lambda f: f(a=1) == f(a=1) == 0),
    )
    assert "lambda **kwargs: 0" in out


def test_impure_function_with_varying_results_falls_back_to_call_log():
    out = _falsifying_output(
        functions(like=func_one_arg, returns=integers(0, 9), pure=False),
        fails(lambda f: f(1) != f(1)),
    )
    assert "lambda" not in out
    assert "Called function: func_one_arg(1) -> 0" in out
    assert "Called function: func_one_arg(1) -> 1" in out


@pytest.mark.parametrize(
    "returns, expected",
    [
        (st.just(3), "lambda x: 3"),
        (st.none(), "lambda x: None"),
        (st.sampled_from([7]), "lambda x: 7"),
        (st.just(-5).map(abs), "lambda x: 5"),
    ],
)
@pytest.mark.parametrize("pure", [False, True])
def test_single_return_value_shown_as_constant(returns, expected, pure):
    # One distinct return value, so there is no point showing a lookup table -
    # note that we detect this by comparison, not by inspecting the strategy.
    out = _falsifying_output(
        functions(like=lambda x: None, returns=returns, pure=pure),
        always_fails(lambda f: (f(0), f(1))),
    )
    assert expected in out


@pytest.mark.parametrize(
    "like, call, expected",
    [
        (func_c, lambda f: f(a=1), "lambda **kwargs: 3"),
        (lambda *a: None, lambda f: f(1), "lambda *a: 3"),
        (lambda: None, lambda f: f(), "lambda: 3"),
    ],
)
def test_constant_shown_whatever_the_signature(like, call, expected):
    out = _falsifying_output(
        functions(like=like, returns=st.just(3), pure=False),
        always_fails(call),
    )
    assert expected in out


class Incomparable:
    # Like a numpy array, __eq__ does not return a bool
    def __eq__(self, other):
        raise ValueError("elementwise comparison")

    __hash__ = None


def test_incomparable_results_fall_back_to_call_log():
    out = _falsifying_output(
        functions(like=func_one_arg, returns=st.builds(Incomparable), pure=False),
        always_fails(lambda f: (f(1), f(2))),
    )
    assert "lambda" not in out
    assert out.count("Called function: func_one_arg") == 2


def test_single_call_is_shown_as_constant():
    # Only one set of inputs, so a lookup table would be needless noise.
    out = _falsifying_output(
        functions(like=lambda x: None, returns=integers(0, 9), pure=True),
        fails(lambda f: f(1) == 0),
    )
    assert "lambda x: 0" in out


def test_functions_supports_find():
    f = find(
        st.functions(like=pure_func, returns=st.integers(), pure=True), lambda x: True
    )
    with pytest.raises(InvalidState):
        f(1, 2)
    assert f.__name__ == pure_func.__name__
