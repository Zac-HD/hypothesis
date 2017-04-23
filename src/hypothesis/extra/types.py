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
    Dict, FrozenSet, Generator, Iterable, Iterator, List, Mapping,
    Sequence, Set, Tuple, TypeVar, Union
)

import hypothesis.strategies as st
from hypothesis.errors import InvalidArgument


StratLookup = Dict[type, st.SearchStrategy]

class ResolutionFailed(TypeError, InvalidArgument):
    """Raised when a type cannot be resolved to a strategy."""
    pass


def resolve(thing: typing.Any, lookup: StratLookup=None) -> st.SearchStrategy:
    # DRAFT ONLY
    """Resolve a type or annotated function to an appropriate search strategy.

    This function basically does dispatch to more specialised resolvers,
    but the generality is very useful when resolving recursively.

    1. If ``thing`` is a type that can be drawn from builtin strategies
       (plus datetime and numpy extra strategies if available), or is in the
       ``lookup`` mapping you supply, the corresponding strategy is returned.
       Note that you can put any key in the lookup mapping, not just types -
       allowing you to e.g. override the strategy chosen for a function.

    2. If ``thing`` is a generic type from the typing module, the resolution
       logic is similar.  As these types are typically specialised (e.g.
       ``List[int]`` -> ``lists(integers())``), no further customisation is
       possible for these types.

    3. If ``thing`` is an function, each argument will be resolved to a
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
    # First, try looking up the type in our mapping of things to strategies
    if thing in type_strategy_mapping(lookup):
        return type_strategy_mapping(lookup)[thing]

    # Any, Union, Optional, TypeVar, etc - 'special types'
    if thing is typing.Any:
        # Any is a particularly special case
        # Is this really the right way to handle it?  (if not fix TypeVar too)
        class CouldBeAnything(object):
            pass
        return st.builds(CouldBeAnything)

    if issubclass(thing, Union):
        return st.one_of([resolve(t) for t in thing.__union_params__])
    if isinstance(thing, TypeVar) and thing.__constraints__:
        # A constrained TypeVar is like a Union of the constraining types
        return st.one_of([resolve(t) for t in thing.__constraints__])
    if isinstance(thing, typing.GenericMeta):
        # Split out into it's own (large) function
        return strategy_from_generic_type(thing)

    raise ResolutionFailed("Could not find strategy for type %s" % thing)


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
        # Built-in collection types don't know their contents, hence are empty
        tuple: st.tuples(),
        list: st.lists(st.nothing()),
        set: st.sets(st.nothing()),
        frozenset: st.frozensets(st.nothing()),
        dict: st.dictionaries(st.nothing(), st.nothing()),
    }
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
    return collections.ChainMap(dict(), known_type_strats, lookup or dict())


def strategy_from_generic_type(thing):
    # DRAFT ONLY
    """Resolve generic types from the typing module to a strategy.

    The goal is for this to handle all such types in the standard library.

    TODO: implement the rest.  Work out how to test that it is in fact
          exhaustive.  See typing.__all__

    """
    assert isinstance(thing, typing.GenericMeta)
    # TODO: all of these assume that the dunder attribute is not None,
    #       i.e. that we're seeing List[int] rather than plain List.
    #       This is not always the case and should be fixed.
    if issubclass(thing, Tuple):
        if hasattr(thing, '_field_types'):
            # it's a typing.NamedTuple
            strats = tuple(resolve(thing._field_types[k])
                           for k in thing.fields)
            return st.builds(thing, *strats)
        if hasattr(thing, '__tuple_params__'):
            # we're dealing with a typing.Tuple[something]
            elem_types = thing.__tuple_params__
            if thing.__tuple_use_ellipsis__:
                return st.lists(resolve(elem_types[0])).map(tuple)
            return st.tuples(*map(resolve, elem_types))
        raise InvalidArgument('No strategy found')  # pragma: no cover
    if issubclass(thing, (List, Sequence, typing.Reversible)):
        return st.lists(resolve(thing.__args__[0]))
    if issubclass(thing, Set):
        return st.sets(resolve(thing.__args__[0]))
    if issubclass(thing, FrozenSet):
        return st.frozensets(resolve(thing.__args__[0]))
    if issubclass(thing, Dict) or issubclass(thing, Mapping):
        keys, vals = (resolve(t) for t in thing.__args__)
        return st.dictionaries(keys, vals)
    if issubclass(thing, (Iterator, Iterable)):
        # TODO: use iterables strategy once merged
        return st.lists(resolve(thing.__args__[0])).map(iter)
    if issubclass(thing, Generator):
        yieldtype, sendtype, returntype = thing.__args__
        def to_gen(alist):
            for val in alist:
                _ = yield val
            # TODO: proper drawing here so it minimises
            return resolve(returntype).example()
        return st.lists(resolve(yieldtype)).map(to_gen)
    if issubclass(thing, typing.Callable):
        return resolve(thing.__result__).map(lambda v: (lambda *_: v))

    # TODO: support the rest of typing.__all__
    raise ResolutionFailed('{} cannot be resolved to a type'.format(thing))


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
    """Fiddly but I think fairly complete?"""
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

