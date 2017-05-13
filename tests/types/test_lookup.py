# coding=utf-8
#
# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis-python
#
# Most of this work is copyright (C) 2013-2017 David R. MacIver
# (david@drmaciver.com), but it contains contributions by others. See
# CONTRIBUTING.rst for a full list of people who may hold copyright, and
# consult the git log if you need to determine who owns an individual
# contribution.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.
#
# END HEADER

"""
Test that the type -> strategy lookup works for the types that it thinks it
can handle.

"""

from __future__ import division, print_function, absolute_import

import typing

import pytest

from hypothesis import given
from hypothesis.extra import types


@pytest.mark.parametrize('typ', types.AllTypingClasses)
def test_resolve_typing_module(typ):
    # TODO: new test which includes parameters, ie List[...] or Dict[..., ...]

    strategy = types.resolve(typ)
    if strategy.is_empty:
        assert issubclass(typ, (typing.Union, typing.TypeVar))
        pytest.skip()

    @given(strategy)
    def inner(ex):
        try:
            # Hackish way to check for _TypeAlias before it explodes below
            issubclass(typ, type)
        except TypeError:
            return
        if typ is typing.Any or isinstance(typ, typing.TypeVar) \
                or typ is typing.Type:
            pass
        elif ex is None and issubclass(typ, typing.Optional):
            assert True
        elif getattr(typ, '_is_protocol', False):
            assert all(hasattr(ex, att) for att in typ.__abstractmethods__)
        elif typ is typing.Tuple:
            assert isinstance(ex, tuple)
        else:
            assert isinstance(ex, typ)

    inner()


@pytest.mark.parametrize('typ', types.type_strategy_mapping())
def test_resolve_concrete_types(typ):
        @given(types.resolve(typ))
        def inner(ex):
            assert isinstance(ex, typ)

        inner()
