# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from inspect import Parameter
from weakref import WeakKeyDictionary

from hypothesis.control import cleanup, should_note
from hypothesis.errors import InvalidState
from hypothesis.internal.reflection import (
    get_signature,
    nicerepr,
    proxies,
    repr_call,
)
from hypothesis.reporting import report
from hypothesis.strategies._internal.strategies import RecurT, SearchStrategy
from hypothesis.vendor.pretty import RepresentationPrinter


def _all_equal(values):
    try:
        return all(v == values[0] for v in values)
    except Exception:
        # Values need not be comparable, and __eq__ can return a non-bool
        # (e.g. numpy arrays) - in which case we just don't summarise.
        return False


class FunctionStrategy(SearchStrategy):
    def __init__(self, like, returns, pure):
        super().__init__()
        self.like = like
        self.returns = returns
        self.pure = pure
        # Both are keyed by generated function; using weakrefs means that they
        # can be garbage-collected at the end of each example, reducing memory
        # use.  Values are {normalised args: return value} for the pure-function
        # cache, and [(repr of call, return value), ...] for the call log.
        self._cache = WeakKeyDictionary()
        self._calls = WeakKeyDictionary()
        self._classified = False

    def _classify(self):
        if self._classified:
            return
        self._signature = get_signature(self.like, follow_wrapped=False)
        self._params = list(self._signature.parameters.values())
        # We can show a pure function as a lambda which looks its arguments up in
        # the mapping of observed calls, unless it takes **kwargs - that dict is
        # unhashable, so the lookup expression would have to flatten it to sorted
        # pairs, which is too ugly to be worth reading.
        self._can_look_up = self.pure and not any(
            p.kind is Parameter.VAR_KEYWORD for p in self._params
        )
        self._classified = True

    def _cache_key(self, args, kwargs):
        # Normalise the calling convention, so that identical arguments give the
        # same key however they were passed.  **kwargs contribute a name-value
        # pair per entry, sorted by name; *args contributes its tuple as-is.
        bound = self._signature.bind(*args, **kwargs)
        bound.apply_defaults()
        key = []
        for p in self._params:
            if p.kind is Parameter.VAR_KEYWORD:
                key.extend(sorted(bound.arguments[p.name].items()))
            else:
                key.append(bound.arguments[p.name])
        return tuple(key)

    def calc_is_empty(self, recur: RecurT) -> bool:
        return recur(self.returns)

    def do_draw(self, data):
        self._classify()
        logged = False

        @proxies(self.like)
        def inner(*args, **kwargs):
            nonlocal logged
            if data.frozen:
                raise InvalidState(
                    f"This generated {nicerepr(self.like)} function can only "
                    "be called within the scope of the @given that created it."
                )
            if self.pure:
                key = self._cache_key(args, kwargs)
                cache = self._cache.setdefault(inner, {})
                new = key not in cache
                if new:
                    cache[key] = data.draw(self.returns)
                val = cache[key]
            else:
                new = True
                val = data.draw(self.returns)

            # optimization to avoid needless repr_call
            if new and should_note():
                rep = repr_call(self.like, args, kwargs, reorder=False)
                self._calls.setdefault(inner, []).append((rep, val))
                if not logged:
                    # Report once, after all calls, so that we can summarise the
                    # complete behaviour of the function instead of each call.
                    logged = True
                    cleanup(lambda: report(self._describe(inner)))
            return val

        return inner

    def _describe(self, inner):
        calls = self._calls[inner]
        values = [val for _, val in calls]
        printer = RepresentationPrinter()
        if _all_equal(values):
            # Only one distinct return value - which covers both a single call
            # and a `returns` strategy with only one element - so show a
            # constant function rather than a lookup which is always the same.
            header = self._lambda_header()
            printer.text(f"lambda {header}: " if header else "lambda: ")
            printer.pretty(values[0])
        elif self._can_look_up:
            self._pretty_lookup(inner, printer)
        else:
            return "\n".join(f"Called function: {rep} -> {val!r}" for rep, val in calls)
        return printer.getvalue()

    def _lambda_header(self):
        parts = []
        star_added = False
        for i, p in enumerate(self._params):
            if p.kind is Parameter.POSITIONAL_ONLY:
                parts.append(p.name)
                if (
                    i + 1 == len(self._params)
                    or self._params[i + 1].kind is not Parameter.POSITIONAL_ONLY
                ):
                    parts.append("/")
            elif p.kind is Parameter.VAR_POSITIONAL:
                parts.append("*" + p.name)
                star_added = True
            elif p.kind is Parameter.KEYWORD_ONLY:
                if not star_added:
                    parts.append("*")
                    star_added = True
                parts.append(p.name)
            elif p.kind is Parameter.VAR_KEYWORD:
                parts.append("**" + p.name)
            else:
                parts.append(p.name)
        return ", ".join(parts)

    def _pretty_lookup(self, inner, p):
        single = len(self._params) == 1
        p.text(f"lambda {self._lambda_header()}: ")
        with p.group(1, "{", "}"):
            for i, (key, value) in enumerate(self._cache[inner].items()):
                if i:
                    p.text(",")
                    p.breakable()
                p.pretty(key[0] if single else key)
                p.text(": ")
                p.pretty(value)
        key_expr = (
            self._params[0].name
            if single
            else "(" + ", ".join(prm.name for prm in self._params) + ")"
        )
        p.text(f"[{key_expr}]")
