RELEASE_TYPE: patch

This patch fixes a regression where :func:`~hypothesis.strategies.recursive` with ``max_leaves`` stopped generating nested structures, causing tests to fail with "Unsatisfiable" errors after the addition of unconditional event realization for solver-based backends (:issue:`4638`).

The fix makes event realization conditional, applying it only when needed (when observability is enabled or when using solver-based backends), avoiding disruption to internal state during recursive strategy retry loops.
