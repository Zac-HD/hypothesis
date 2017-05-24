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

import functools
import collections
from inspect import isclass

import hypothesis.strategies as st
from hypothesis.errors import InvalidArgument
from hypothesis.internal.compat import text_type


def from_type(thing, lookup=None):
    """Resolve a type to an appropriate search strategy.

    **Import this function from hypothesis.strategies, not here.**

    1. If ``thing`` is a subclass of something from the typing module, the
       corresponding strategy is returned.  For abstract types this may be
       the union of one or more strategies for concrete subtypes.

    2. If ``thing`` is a type that can be drawn from a builtin strategy or
       an importable extra strategy, or is in the ``lookup`` mapping you
       supply, the corresponding strategy is returned.  If there is no exact
       match, do a subtype lookup in the chained mappings.

    The subtype lookup walks down the inheritance tree, adding each strategy
    it finds to the mix and ignoring things further down that branch, e.g.::

        List[int] -> lists(elements=integers())
        int       -> integers()
        Sequence  -> lists() | tuples()

    """
    # Look for a known concrete type or user-defined mapping
    lookup = type_strategy_mapping(lookup)
    if thing in lookup:
        return lookup[thing]
    # I tried many more elegant checks, but `typing` tends to treat the type
    # system as a loose guideline at best so they were all unreliable.
    if getattr(thing, '__module__', None) == 'typing':
        return from_typing_type(thing)
    # If there's no exact match above, use similar subtype resolution logic
    lookup = {k: v for k, v in lookup.items() if isclass(k)}
    return st.one_of([
        v for k, v in lookup.items()
        if issubclass(k, thing) and sum(issubclass(k, T) for T in lookup) == 1
    ])


@st.cacheable
def type_strategy_mapping(lookup=None):
    """Return a dict mapping from types to corresponding search strategies.

    Most resolutions will terminate here or in the special handling for
    generics from the typing module.

    """
    import uuid
    import decimal
    import datetime as dt
    import fractions

    known_type_strats = {
        type(None): st.none(),
        bool: st.booleans(),
        int: st.integers(),
        float: st.floats(),
        complex: st.complex_numbers(),
        fractions.Fraction: st.fractions(),
        decimal.Decimal: st.decimals(),
        text_type: st.characters() | st.text(),
        bytes: st.binary(),
        dt.datetime: st.datetimes(),
        dt.date: st.dates(),
        dt.time: st.times(),
        uuid.UUID: st.uuids(),
    }
    # build empty collections, as only generics know their contents
    known_type_strats.update({
        t: st.builds(t) for t in (tuple, list, set, frozenset, dict)
    })
    try:
        from hypothesis.extra.pytz import timezones
        known_type_strats[dt.tzinfo] = timezones()
    except ImportError:
        pass
    try:
        import numpy as np
        from hypothesis.extra.numpy import \
            arrays, array_shapes, scalar_dtypes, nested_dtypes
        known_type_strats[np.dtype] = nested_dtypes()
        known_type_strats[np.ndarray] = arrays(
            scalar_dtypes(), array_shapes(max_dims=2))
    except ImportError:
        pass
    known_type_strats.update(lookup or {})
    return known_type_strats


def get_all_typing_classes():
    try:
        import typing
    except ImportError:
        return ()
    return tuple(
        cls for cls in (getattr(typing, name) for name in typing.__all__)
        if isclass(cls) and cls is not typing.Generic
        and not isinstance(cls, typing._ProtocolMeta)
    ) + (typing.BinaryIO, typing.TextIO, typing.re.Pattern, typing.re.Match)


def from_typing_type(thing):
    try:
        import typing
    except ImportError:
        return st.nothing()
    # `Any` and `Type` mess up our subclass lookups, so handle them first
    if thing is typing.Any:
        # TODO: this is notionally correct, but also entirely broken - see
        # https://github.com/HypothesisWorks/hypothesis-python/issues/491
        import unittest
        return st.builds(unittest.mock.MagicMock)
    if thing is typing.Type:
        if thing.__args__ is None:
            return st.just(type)
        return st.just(thing.__args__[0])

    def try_issubclass(thing, maybe_superclass):
        try:
            return issubclass(thing, maybe_superclass)
        except TypeError:
            # TODO:  upstream report:  issubclass(x, typing._TypeAlias) fails
            # when it reaches collections.abc, because it can't be weakrefed
            return None

    # Of all types with a strategy, select the supertypes of this thing that
    # whose subtypes have no strategy, and return their strategic union
    mapping = {k: v for k, v in generic_type_strategy_mapping().items()
               if try_issubclass(k, thing)}
    return st.one_of([v(thing) for k, v in mapping.items()
                      if sum(issubclass(k, T) for T in mapping) == 1])


@st.composite
def generators(draw, yield_strat, ret_strat=st.none()):

    def to_gen(alist, retval):
        for val in alist:
            _ = yield val
            _
        return retval

    return to_gen(draw(st.lists(yield_strat)), draw(ret_strat))


def nary_callable(args, retval):
    # Placeholder strategy, to be replaced with a better version later.
    # See https://github.com/HypothesisWorks/hypothesis-python/issues/167
    if args is None:
        return lambda: retval
    if args is Ellipsis:
        return lambda *_: retval
    # TODO: handle more detailed argspec enabled by extended callable types
    # See https://mypy-lang.blogspot.com.au/2017/05/mypy-0510-released.html
    args = ', '.join('_arg' + str(n) for n in range(len(args)))
    # See https://github.com/HypothesisWorks/hypothesis-python/issues/387
    return eval('lambda %s: retval' % args)


class AsyncIteratorWrapper:
    # based on example code from https://www.python.org/dev/peps/pep-0492/
    # with extra workarounds to run under py2 and work under 3.4
    def __init__(self, obj):
        self._obj = obj
        self._it = iter(obj)

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self._obj)

    def __aiter__(self):
        return self

    try:
        from asyncio import coroutine
    except ImportError:
        def coroutine(f):
            return f

    @coroutine
    def __anext__(self):
        try:
            value = next(self._it)
        except StopIteration:
            try:
                raise StopAsyncIteration
            except NameError:  # pragma: no cover
                raise Exception
        return value


@st.cacheable
def generic_type_strategy_mapping():
    """Cache most of our generic type resolution logic.

    Requires the ``typing`` module to be importable.

    """
    import io
    import re
    import typing

    # TODO: work out why io.StringIO is not a typing.io.TextIO thing
    # TODO: likewise BinaryIO

    registry = {
        # Some types are not generic, so we can write the lookup immediately
        typing.ByteString: lambda _: st.binary(),
        typing.io.BinaryIO: lambda _: st.builds(io.BytesIO, st.binary()),
        typing.io.TextIO: lambda _: st.builds(io.StringIO, st.text()),
        # TODO:  strategy for generating valid regex patterns
        typing.re.Match[text_type]: (
            lambda _: st.builds(lambda s: re.match(u'.*', s), st.binary())
        ),
        typing.re.Match[bytes]: (
            lambda _: st.builds(lambda s: re.match(b'.*', s), st.binary())
        ),
        typing.re.Pattern[text_type]: lambda _: st.just(re.compile(u'.*')),
        typing.re.Pattern[bytes]: lambda _: st.just(re.compile(b'.*')),
    }

    def register(type_, fallback=None, attr='__args__'):
        def inner(func):
            if fallback is None:
                registry[type_] = func
                return func
            @functools.wraps(func)
            def really_inner(thing):
                if getattr(thing, attr, None) is None:
                    return fallback
                return func(thing)
            registry[type_] = really_inner
            return really_inner
        return inner

    @register(typing.Union)
    def resolve_Union(thing):
        possible = (getattr(thing, '__union_params__', None) or ()) + (
            getattr(thing, '__args__', None) or ())
        return st.one_of([from_type(t) for t in possible])

    @register(typing.Optional)
    def resolve_Optional(thing):
        return st.none() | resolve_Union(thing)

    @register(typing.TypeVar)
    def resolve_TypeVar(thing):
        if getattr(thing, '__contravariant__', False):
            raise InvalidArgument('Cannot resolve contravariant %s' % thing)
        return st.one_of([from_type(t) for t in
                          getattr(thing, '__constraints__', ())])

    @register(typing.AnyStr)
    def resolve_AnyStr(thing):
        return st.one_of([from_type(t) for t in typing.AnyStr.__constraints__])

    @register(typing.Tuple)
    def resolve_Tuple(thing):
        if hasattr(thing, '_field_types'):
            # it's a typing.NamedTuple
            strats = tuple(from_type(thing._field_types[k])
                           for k in thing.fields)
            return st.builds(thing, *strats)
        if hasattr(thing, '__tuple_params__'):
            # we're dealing with a typing.Tuple[something]
            elem_types = thing.__tuple_params__
            if elem_types is None:
                return st.tuples()
            if thing.__tuple_use_ellipsis__:
                return st.lists(from_type(elem_types[0])).map(tuple)
            return st.tuples(*map(from_type, elem_types))

    @register(typing.Callable, st.just(lambda: None))
    def resolve_Callable(thing):
        return from_type(thing.__result__).map(
            functools.partial(nary_callable, thing.__args__))

    @register(typing.List, st.builds(list))
    def resolve_List(thing):
        return st.lists(from_type(thing.__args__[0]))

    @register(typing.Set, st.builds(set))
    def resolve_Set(thing):
        return st.sets(from_type(thing.__args__[0]))

    @register(typing.FrozenSet, st.builds(frozenset))
    def resolve_FrozenSet(thing):
        return st.frozensets(from_type(thing.__args__[0]))

    @register(typing.Dict, st.builds(dict))
    def resolve_Dict(thing):
        keys, vals = (from_type(t) for t in thing.__args__)
        return st.dictionaries(keys, vals)

    @register(typing.DefaultDict, st.builds(collections.defaultdict))
    def from_type(thing):
        keys, vals = (from_type(t) for t in thing.__args__)
        return st.dictionaries(keys, vals).map(
            lambda d: collections.defaultdict(None, d))

    @register(typing.ItemsView)
    def resolve_ItemsView(thing):
        return resolve_Dict(thing).map(dict.items)

    @register(typing.KeysView, st.builds(dict).map(dict.keys))
    def resolve_KeysView(thing):
        return st.dictionaries(from_type(thing.__args__[0]), st.none()
                               ).map(dict.keys)

    @register(typing.ValuesView, st.builds(dict).map(dict.values))
    def resolve_ValuesView(thing):
        return st.dictionaries(st.integers(), from_type(thing.__args__[0])
                               ).map(dict.values)

    @register(typing.Iterator, st.iterables(st.nothing()))
    def resolve_Iterator(thing):
        return st.iterables(from_type(thing.__args__[0]))

    @register(typing.AsyncIterator)
    def resolve_AsyncIterator(thing):
        return resolve_Iterator(thing).map(AsyncIteratorWrapper)

    @register(typing.Awaitable)
    def resolve_Awaitable(thing):
        return resolve_AsyncIterator(thing).map(lambda ai: ai.__anext__())

    @register(typing.Generator, generators(st.nothing()))
    def resolve_Generator(thing):
        yieldtype, _, returntype = thing.__args__
        return generators(from_type(yieldtype), from_type(returntype))

    @register(typing.re.Match)
    def resolve_re_Match(_):
        return st.one_of(
            st.builds(lambda s: re.match(u'.*', s), st.text()),
            st.builds(lambda s: re.match(b'.*', s), st.binary())
        )

    return registry
