# coding=utf-8
#
# Copyright (C) 2017 Zac Hatfield-Dodds
# Shared under the terms of the GNU Affero GPL licence version 3+
#
# I indend to relicence under the MPL2 if this can be merged into Hypothesis,
# but will otherwise release as a seperate package under strong copyleft.
# Alternative licenses negotiable.

"""
Create strategies for types or annotated functions
==================================================

As suggested by the title, this module supplies functions that do two things:

1. Takes a type, and returns an appropriate strategy
2. Takes an annotated function, and acts like an automatic ``builds``


Useful Links
------------

- https://github.com/HypothesisWorks/hypothesis-python/issues/293
- https://docs.python.org/3/library/typing.html


Requirements
------------
(See linked issue)

- Users must be able to define additional type: strategy relationships
- Users must be able to override specific parts of the lookup, to avoid a
  big jump between automated and manual strategy definition

The latter will be particularly annoying.

"""

from __future__ import division, print_function, absolute_import

import collections
import contextlib
import decimal
import fractions
import functools
import inspect
import string
import typing
from typing import (
    Dict, FrozenSet, Generator, Iterator, List, Set, Tuple, TypeVar, Union
)
from unittest import mock

import hypothesis.strategies as st
from hypothesis.errors import InvalidArgument


StratLookup = Dict[type, st.SearchStrategy]
# TODO: use this to test that we can resolve all of them
AllTypingClasses = tuple(
    cls for cls in (getattr(typing, name) for name in typing.__all__)
    if inspect.isclass(cls) and cls is not typing.Generic
)


class ResolutionFailed(InvalidArgument):
    """Raised when a type cannot be resolved to a strategy."""
    pass


def resolve(thing: typing.Any, lookup: StratLookup=None) -> st.SearchStrategy:
    # DRAFT ONLY
    """Resolve a type to an appropriate search strategy.

    1. If ``thing`` is a subclass of something from the typing module, the
       corresponding strategy is returned.  Note that while concrete types
       have exact strategies, abstract types resolve to some combination of
       concrete strategies.

    2. If ``thing`` is a type that can be drawn from a builtin strategy or
       an importable extra strategy, or is in the ``lookup`` mapping you
       supply, the corresponding strategy is returned.  If there is no exact
       match, do a subtype lookup in the chained mappings.

    The subtype lookup walks down the inheritance tree, adding each strategy
    it finds to the mix and ignoring things further down that branch, e.g.::

        List[int] -> lists(elements=integers())
        int       -> integers()
        Sequence  -> lists() | tuples()

    TODO: possibly introspect and see if `builds` would work as a fallback?

    """
    if getattr(thing, '__module__', None) == 'typing':
        if thing is typing.Any:
            return st.builds(mock.MagicMock)
        if thing is typing.Type:
            if thing.__args__ is None:
                return st.just(type)
            return st.just(thing.__args__[0])

        mapping = {k: v for k, v in generic_type_strategy_mapping().items()
                  if issubclass(thing, k)}
        strat = st.one_of([v(thing) for k, v in mapping.items()
                           if sum(issubclass(k, T) for T in mapping) == 1])
        if not strat.is_empty:
            return strat
    lookup = type_strategy_mapping(lookup)
    if thing in lookup:
        return lookup[thing]
    lookup = {k: v for k, v in lookup.items() if inspect.isclass(k)}
    strat = st.one_of([
        v for k, v in lookup.items()
        if issubclass(k, thing) and sum(issubclass(k, T) for T in lookup) == 1
    ])
    if not strat.is_empty:
        return strat
    raise ResolutionFailed('Could not find strategy for type %r' % thing)


@functools.lru_cache()
def type_strategy_mapping(lookup: StratLookup=None) -> StratLookup:
    """Return a mapping from types to corresponding search strategies.

    Most resolutions will terminate here or in the special handling for
    generics from the typing module.

    """
    known_type_strats = {
        type(None): st.none(),
        bool: st.booleans(),
        int: st.integers(),
        float: st.floats(),
        complex: st.complex_numbers(),
        fractions.Fraction: st.fractions(),
        decimal.Decimal: st.decimals(),
        str: st.characters() | st.text(),
        bytes: st.binary(),
    }
    # build empty collections, as only generics know their contents
    known_type_strats.update({
        t: st.builds(t) for t in (tuple, list, set, frozenset, dict)
    })
    # TODO: add the equivalent entry for extra.django - model lookup??
    with contextlib.suppress(ImportError):
        import datetime as dt
        import  hypothesis.extra.datetime as dt_strats
        known_type_strats.update({
            dt.datetime: dt_strats.datetimes(),
            dt.date: dt_strats.dates(),
            dt.time: dt_strats.times(),
            # TODO: add timedeltas once pull is merged
        })
    with contextlib.suppress(ImportError):
        import numpy as np
        from hypothesis.extra.numpy import \
            arrays, array_shapes, scalar_dtypes, nested_dtypes
        known_type_strats.update({
            np.ndarray: arrays(scalar_dtypes(), array_shapes(max_dims=2)),
            np.dtype: nested_dtypes(),
        })
    if lookup:
        known_type_strats.update(lookup)
    return known_type_strats


@st.composite
def generators(draw, yield_strat, ret_strat=None):

    def to_gen(alist):
        for val in alist:
            _ = yield val
            _
        if ret_strat is not None:
            return draw(ret_strat)
    return draw(st.lists(yield_strat).map(to_gen))


def nary_callable(args, retval):
    if args is None:
        return lambda: retval
    if args is Ellipsis:
        return lambda *_: retval
    # TODO: handle more detailed argspec enabled by extended callable types
    # See https://mypy-lang.blogspot.com.au/2017/05/mypy-0510-released.html
    args = ', '.join('_arg' + str(n) for n in range(len(args)))
    return eval('lambda %s: retval' % args)


class AsyncIteratorWrapper:
    # based on https://www.python.org/dev/peps/pep-0492/
    def __init__(self, obj):
        self._obj = obj
        self._it = iter(obj)

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self._obj)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            value = next(self._it)
        except StopIteration:
            raise StopAsyncIteration
        return value


@functools.lru_cache()
def generic_type_strategy_mapping():
    """Cache most of our generic type resolution logic.

    :py:func:`resolve` does the heavy lifting, but here we supply a dictionary
    with generic types as keys and a function to resolve that type as values.

    """
    registry = {
        typing.ByteString: lambda _: st.binary(),
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

    @register(Union)
    def resolve_Union(thing):
        return st.one_of([resolve(t) for t in
                          getattr(thing, '__union_params__', None) or ()])

    @register(TypeVar)
    def resolve_TypeVar(thing):
        return st.one_of([resolve(t) for t in
                          getattr(thing, '__constraints__', ())])

    @register(Tuple)
    def resolve_Tuple(thing):
        if hasattr(thing, '_field_types'):
            # it's a typing.NamedTuple
            strats = tuple(resolve(thing._field_types[k])
                           for k in thing.fields)
            return st.builds(thing, *strats)
        if hasattr(thing, '__tuple_params__'):
            # we're dealing with a typing.Tuple[something]
            elem_types = thing.__tuple_params__
            if elem_types is None:
                return st.tuples()
            if thing.__tuple_use_ellipsis__:
                return st.lists(resolve(elem_types[0])).map(tuple)
            return st.tuples(*map(resolve, elem_types))

    @register(typing.Callable, st.just(lambda: None))
    def resolve_Callable(thing):
        return resolve(thing.__result__).map(
            functools.partial(nary_callable, thing.__args__))

    @register(List, st.builds(list))
    def resolve_List(thing):
        return st.lists(resolve(thing.__args__[0]))

    @register(Set, st.builds(set))
    def resolve_Set(thing):
        return st.sets(resolve(thing.__args__[0]))

    @register(FrozenSet, st.builds(frozenset))
    def resolve_FrozenSet(thing):
        return st.frozensets(resolve(thing.__args__[0]))

    @register(Dict, st.builds(dict))
    def resolve_Dict(thing):
        keys, vals = (resolve(t) for t in thing.__args__)
        return st.dictionaries(keys, vals)

    @register(typing.DefaultDict, st.builds(collections.defaultdict))
    def resolve_DefaultDict(thing):
        keys, vals = (resolve(t) for t in thing.__args__)
        return st.dictionaries(keys, vals).map(
            lambda d: collections.defaultdict(None, d))

    @register(typing.ItemsView)
    def resolve_ItemsView(thing):
        return resolve_Dict(thing).map(dict.items)

    @register(typing.KeysView, st.builds(dict).map(dict.keys))
    def resolve_KeysView(thing):
        return st.dictionaries(resolve(thing.__args__[0]), st.none()
                               ).map(dict.keys)

    @register(typing.ValuesView, st.builds(dict).map(dict.values))
    def resolve_ValuesView(thing):
        return st.dictionaries(st.integers(), resolve(thing.__args__[0])
                               ).map(dict.values)

    @register(Iterator, st.iterables(st.nothing()))
    def resolve_Iterator(thing):
        return st.iterables(resolve(thing.__args__[0]))

    @register(typing.AsyncIterator)
    def resolve_AsyncIterator(thing):
        return resolve_Iterator(thing).map(AsyncIteratorWrapper)

    @register(typing.Awaitable)
    def resolve_Awaitable(thing):
        return resolve_AsyncIterator(thing).map(lambda ai: ai.__anext__())

    @register(Generator, generators(st.nothing()))
    def resolve_Generator(thing):
        yieldtype, _, returntype = thing.__args__
        return generators(resolve(yieldtype), resolve(returntype))

    return registry


def check_all_annotated(thing, lookup):
    # DRAFT ONLY
    args = set(inspect.signature(thing).parameters) | {'return'}
    types = typing.get_type_hints(thing)
    missing = sorted(args - set(types))
    if missing:
        raise InvalidArgument(
            '{} has unannotated arguments or return type: {}'
            .format(thing, missing))
    no_strat = set(types.values()) - (
        set(lookup) | set(type_strategy_mapping()))
    if no_strat:
        raise InvalidArgument(
            'No known strategy for types {} in {}'.format(no_strat, thing))


@st.composite
def from_type(draw, thing, lookup=None):
    # DRAFT ONLY
    """Fiddly but I think fairly complete?

    If ``thing`` is an function, each argument will be resolved to a
    strategy.  The returned strategy draws a value for each argument,
    calls the function, and returns the result.  All arguments must have
    either a resolvable type annotation or a default value.

    TODO: or possibly a strategy over zero-argument callables, to force
          exception handling back to the user?  Could store the arguments
          on an attribute too; useful for eg round-trip tests...

    Note that while types are used to infer the appropriate strategy,
    Hypothesis will not check that returned values (etc) match the declared
    types - you will need to write those tests for yourself.

    """
    # TODO: fix lookup (broken), add lookup to resolve
    # TODO: factor into a pure lookup function, and an execution helper
    # TODO: integrate into resolve

    lookup = lookup or set()
    if isinstance(thing, type):
        # TODO: should check thing.__init__ in case it's a class before this
        return draw(resolve(thing, lookup))
    check_all_annotated(thing, lookup)
    arg_types = dict(typing.get_type_hints(thing))
    ret_type = arg_types.pop('return')
    spec = inspect.getfullargspec(thing)
    varargs, varkw = (), dict()
    if spec.varargs is not None:
        strat = from_type(arg_types.pop(spec.varargs), lookup)
        varargs = draw(st.lists(strat))
    if spec.varkw is not None:
        keys_strat = st.text('_' + string.ascii_letters, min_size=1)
        vals_strat = from_type(arg_types.pop(spec.varkw), lookup)
        varkw = draw(st.dictionaries(keys_strat, vals_strat))
    other_args = {k: draw(from_type(v)) for k, v in arg_types.items()}
    ret_val = thing(*varargs, **varkw, **other_args)
    if not isinstance(ret_val, ret_type):
        raise TypeError(
            'Return value {} was not of type {} from {}(*{}, **{}, **{})'
            .format(ret_val, ret_type.__name__, thing.__name__, varargs,
                    varkw, other_args))
    return ret_val

