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

from __future__ import division, print_function, absolute_import

import io
import typing

import pytest

from hypothesis import given
from hypothesis.strategies import from_type
from hypothesis.searchstrategy import types


@pytest.mark.parametrize('typ', types.get_all_typing_classes())
def test_resolve_typing_module(typ):
    # This is covered in special cases; the real purpose is to make sure we
    # notice if something (else) odd is added to the typing module.

    # TODO: new test which includes parameters, ie List[...] or Dict[..., ...]
    try:
        # Hackish way to check for _TypeAlias before it explodes below
        issubclass(typ, type)
    except TypeError:
        pytest.skip()

    strategy = from_type(typ)

    @given(strategy)
    def inner(ex):
        if typ is typing.Any or isinstance(typ, typing.TypeVar) \
                or typ is typing.Type:
            pass
        elif ex is None and issubclass(typ, typing.Optional):
            assert True
        elif getattr(typ, '_is_protocol', False):
            assert all(hasattr(ex, att) for att in typ.__abstractmethods__)
        elif typ is typing.Tuple:
            assert isinstance(ex, tuple)
        elif typ in (typing.BinaryIO, typing.TextIO):
            assert isinstance(ex, io.IOBase)
        else:
            assert isinstance(ex, typ)

    inner()
