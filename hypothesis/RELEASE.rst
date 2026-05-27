RELEASE_TYPE: minor

:func:`~hypothesis.strategies.functions` now mimics the asynchronous and
generator nature of its ``like=`` argument (:issue:`4149`).  If ``like`` is a
coroutine function, generator function, or async generator function, the
generated function will be one too - so that ``await``-ing or iterating over
the result behaves as you'd expect, and the matching :mod:`inspect` predicate
returns ``True``.

For generator and async-generator functions, the ``returns`` argument (or the
element of an inferred yield type such as ``Iterator[int]``) describes a single
yielded value, and each call yields a list of such values.

This is technically a behaviour change for ``like`` callables which were
already coroutine or generator functions, but as the previously-generated plain
functions could not be awaited or iterated we expect the impact to be minimal.
