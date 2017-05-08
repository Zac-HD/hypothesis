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

    try:
        @given(types.resolve(typ))
        def inner(ex):
            if typ is typing.Any or isinstance(typ, typing.TypeVar) \
                    or issubclass(typ, typing._Protocol):
                pass
            elif typ is typing.Tuple:
                assert isinstance(ex, tuple)
            else:
                assert isinstance(ex, typ)

    except types.ResolutionFailed:
        pytest.skip()
    inner()


@pytest.mark.parametrize('typ', types.type_strategy_mapping())
def test_resolve_concrete_types(typ):
        @given(types.resolve(typ))
        def inner(ex):
            assert isinstance(ex, typ)

        inner()
