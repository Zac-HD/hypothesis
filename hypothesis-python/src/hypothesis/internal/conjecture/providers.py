# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import enum
from fractions import Fraction
import hashlib
import heapq
import math
import sys
from collections import OrderedDict, abc
from functools import lru_cache
import time
from typing import (
    TYPE_CHECKING,
    Any,
    List,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)
import warnings

from hypothesis.errors import HypothesisWarning, InvalidArgument
from hypothesis.internal.compat import floor, int_from_bytes
from hypothesis.internal.floats import int_to_float, next_up
from hypothesis.internal.conjecture.data import MAX_DEPTH, ConjectureData
from hypothesis.internal.conjecture import utils as cu

if TYPE_CHECKING:
    from hypothesis.strategies._internal.strategies import Ex, SearchStrategy

ONE_BOUND_INTEGERS_LABEL = cu.calc_label_from_name("trying a one-bound int allowing 0")


@lru_cache(maxsize=1024)
def _get_sampler(weights: Sequence[float]) -> cu.Sampler:
    return cu.Sampler(weights)


class PrimitiveProvider:
    # This is the low-level interface which would also be implemented
    # by e.g. CrossHair, by an Atheris-hypothesis integration, etc.
    # We'd then build the structured tree handling, database and replay
    # support, etc. on top of this - so all backends get those for free.

    def __init__(self, conjecturedata: ConjectureData, /) -> None:
        self._cd = conjecturedata

    def draw_boolean(self, *, p: float = 0.5, forced: Optional[bool] = None):
        # Note that this could also be implemented in terms of draw_integer().
        return cu.biased_coin(self._cd, p=p, forced=forced)

    def draw_integer(
        self,
        *,
        forced: Optional[int] = None,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
        # weights are for choosing an element index from a bounded range
        weights: Optional[Sequence[float]] = None,
        shrink_towards: int = 0,
    ):
        # Validate arguments
        if weights is not None:
            assert min_value is not None
            assert max_value is not None
            assert (max_value - min_value) <= 1024  # arbitrary practical limit
        if forced is not None:
            assert min_value is None or min_value <= forced
            assert max_value is None or forced <= max_value

        # This is easy to build on top of our existing conjecture utils,
        # and it's easy to build sampled_from and weighted_coin on this.
        if weights is not None:
            sampler = _get_sampler(weights)
            assert forced is None  # FIXME: handle forced values here
            idx = sampler.sample(self._cd)
            assert shrink_towards <= min_value  # FIXME: reorder for good shrinking
            return range(min_value, max_value + 1)[idx]

        if min_value is None and max_value is None:
            return cu.unbounded_integers(self._cd, forced=forced)

        if min_value is None:
            if max_value <= shrink_towards:
                return max_value - abs(cu.unbounded_integers(self._cd, forced=forced))
            else:
                probe = max_value + 1
                while max_value < probe:
                    self._cd.start_example(ONE_BOUND_INTEGERS_LABEL)
                    probe = cu.unbounded_integers(self._cd, forced=forced)
                    self._cd.stop_example(discard=max_value < probe)
                return probe

        if max_value is None:
            if min_value >= shrink_towards:
                return min_value + abs(cu.unbounded_integers(self._cd, forced=forced))
            else:
                probe = min_value - 1
                while probe < min_value:
                    self._cd.start_example(ONE_BOUND_INTEGERS_LABEL)
                    probe = cu.unbounded_integers(self._cd, forced=forced)
                    self._cd.stop_example(discard=probe < min_value)
                return probe

        # For bounded integers, make the bounds and near-bounds more likely.
        if max_value - min_value > 127:
            bits = self._cd.draw_bits(7, forced=0 if forced is not None else None)
            forced = {
                122: min_value,
                123: min_value,
                124: max_value,
                125: max_value,
                126: min_value + 1,
                127: max_value - 1,
            }.get(bits, forced)

        return cu.integer_range(
            self._cd,
            min_value,
            max_value,
            center=shrink_towards,
            forced=forced,
        )

    def draw_float(
        self,
        *,
        forced: Optional[float] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        allow_nan: bool = True,
        allow_infinity: bool = True,
        allow_subnormal: bool = True,
        width: Literal[16, 32, 64] = 64,
        # exclude_min and exclude_max handled higher up
    ):
        ...

    def draw_string(
        self,
        *,
        forced: Optional[str] = None,
        # Should we use `regex: str = ".*"` instead of alphabet and sizes?
        alphabet: ... = ...,
        min_size: int = 0,
        max_size: Optional[int] = None,
    ):
        ...

    # def draw_bytes(
    #     self,
    #     *,
    #     forced: Optional[bytes] = None,
    #     min_size: int = 0,
    #     max_size: Optional[int] = None,
    # ):
    #     ...


class UseProviderWarning(HypothesisWarning):
    pass


class Provider:
    """Wraps a PrimitiveProvider and implements all the useful tracking information.

    For example spans, database support, etc. are all implemented at this layer.

    FIXME: we currently pretend that this is a ConjectureData object itself,
           in order to get a more gradual migration pathway.
    """

    def __init__(self, pp: PrimitiveProvider, /) -> None:
        self._pp = pp

        self.draw_boolean = pp.draw_boolean
        self.draw_integer = pp.draw_integer
        self.draw_float = pp.draw_float
        self.draw_string = pp.draw_string
        # self.draw_bytes = pp.draw_bytes

    def __getattr__(self, __name: str) -> Any:
        # FIXME: port callers and then delete this method
        # warnings.warn("use Provider interface", UseProviderWarning, stacklevel=2)
        return getattr(self._pp._cd, __name)

    def draw(self, strategy: "SearchStrategy[Ex]", label: Optional[int] = None) -> "Ex":
        if self.is_find and not strategy.supports_find:
            raise InvalidArgument(
                f"Cannot use strategy {strategy!r} within a call to find "
                "(presumably because it would be invalid after the call had ended)."
            )

        at_top_level = self.depth == 0
        start_time = None
        if at_top_level:
            # We start this timer early, because accessing attributes on a LazyStrategy
            # can be almost arbitrarily slow.  In cases like characters() and text()
            # where we cache something expensive, this led to Flaky deadline errors!
            # See https://github.com/HypothesisWorks/hypothesis/issues/2108
            start_time = time.perf_counter()

        strategy.validate()

        if strategy.is_empty:
            self.mark_invalid("strategy is empty")

        if self.depth >= MAX_DEPTH:
            self.mark_invalid("max depth exceeded")

        if label is None:
            assert isinstance(strategy.label, int)
            label = strategy.label
        self.start_example(label=label)
        try:
            if not at_top_level:
                return strategy.do_draw(self)
            else:
                assert start_time is not None
                strategy.validate()
                try:
                    return strategy.do_draw(self)
                finally:
                    self.draw_times.append(time.perf_counter() - start_time)
        finally:
            self.stop_example()
