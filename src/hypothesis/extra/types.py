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


def resolve(thing: typing.Any, lookup: StratLookup=None) -> st.SearchStrategy:
    # DRAFT ONLY
    """Resolve a type to an appropriate search strategy.

    ``thing`` may be a type, including the generic types from the typing
    module, or a type-annotated function.  Not all values which match a
    type can always be drawn from the strategy; some shortcuts or simpler
    (but still matching) values may be used.  For annotated functions,
    we may fall back to defaults etc. if some annotations cannot be
    resolved.

    This function basically does dispatch to more specialised resolvers,
    when the sheer generality is useful.

    """
    # Try direct type lookup first
    known_types = type_strategy_mapping(lookup)
    if thing in known_types:
        # This returns registered strategies for an exact type.
        # We try subtypes last, after checking special and container types.
        return known_types[thing]

    # Any, Union, Optional, TypeVar, etc - 'special types'
    if thing is typing.Any:
        # Any is a particularly special case
        # Is this really the right way to handle it?  (if not fix TypeVar too)

        class CouldBeAnything(object):
            pass
        return st.builds(CouldBeAnything)

    if issubclass(thing, Union):
        params = thing.__union_params__
        if type(None) in params:
            # Minimise to None for Optional[T]
            params = (type(None),) + tuple(
                t for t in params if t is not type(None))
        return st.one_of([resolve(t) for t in params])

    if isinstance(thing, TypeVar):
        # We can treat a constrained TypeVar as a Union of the constraining
        # types.  An unconstrained typevar should stand in for all types,
        # so we can test it with a subtype of `object`.
        if not thing.__constraints__:
            return resolve(typing.Any)
        return st.one_of([resolve(t) for t in params])

    if isinstance(thing, typing.GenericMeta):
        # Split out into it's own (large) function
        return strategy_from_generic_type(thing)

    # This is the subtype lookup mentioned above.
    subtype_of = [t for t in known_types
                  if isinstance(t, type) and issubclass(thing, t)]
    if subtype_of:
        no_subtypes = [t for t in subtype_of if sum(
            issubclass(st, t) for st in subtype_of) == 1]
        return st.one_of([known_types[t] for t in no_subtypes]).map(thing)

    raise InvalidArgument("Could not find strategy for type %s" % thing)


@functools.lru_cache()
def type_strategy_mapping(lookup: StratLookup=None) -> StratLookup:
    """Return a mapping from types to corresponding search strategies.

    Most resolutions will terminate here or in the special handling for
    generics from the typing module.

    TODO: take a supplementary argument of user-defined type: strategy
          relationships and include them in the lookup somehow.

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
    try:
        import datetime as dt
        import  hypothesis.extra.datetime as dt_strats
        known_type_strats.update({
            dt.datetime: dt_strats.datetimes(),
            dt.date: dt_strats.dates(),
            dt.time: dt_strats.times(),
            # TODO: add timedeltas once pull is merged
        })
    except ImportError:
        pass
    try:
        import numpy as np
        from hypothesis.extra.numpy import \
            arrays, array_shapes, scalar_dtypes, nested_dtypes
        known_type_strats.update({
            np.ndarray: arrays(scalar_dtypes(), array_shapes(max_dims=2)),
            np.dtype: nested_dtypes(),
        })
    except ImportError:
        pass
    return collections.ChainMap(dict(), known_type_strats, lookup or dict())


def strategy_from_generic_type(thing):
    # DRAFT ONLY
    """Resolve generic types from the typing module to a strategy.

    The goal is for this to handle all such types in the standard library.

    TODO: implement the rest.  Work out how to test that it is in fact
          exhaustive.

    """
    assert isinstance(thing, typing.GenericMeta)
    # TODO: all of these assume that the dunder attribute is not None,
    #       i.e. that we're seeign List[int] rather than plain List.
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
    if issubclass(thing, Iterator)  or issubclass(thing, Iterable):
        # TODO: use iterables strategy once merged
        return st.lists(resolve(thing.__args__[0])).map(iter)
    if issubclass(thing, Generator):
        yieldtype, sendtype, returntype = thing.__args__
        def to_gen(alist):
            yield from alist
            # TODO: proper drawing here so it minimises
            return resolve(returntype).example()
        return st.lists(resolve(yieldtype)).map(to_gen)
    if issubclass(thing, typing.Callable):
        n_ary = len(thing.__args__)
        def strat(value):
            def func(*args):
                assert len(args) == n_ary
                return value
            return func
        return resolve(thing.__result__).map(lambda v: (lambda *_: v))

    # TODO: support the rest of typing.__all__
    raise NotImplementedError


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

