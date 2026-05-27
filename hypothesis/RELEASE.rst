RELEASE_TYPE: patch

This patch improves shrinking of tests where an early choice controls the size
of a later collection, such as ``n = data.draw(integers()); s =
data.draw(text(min_size=n, max_size=n))``.  Lowering ``n`` would previously
discard the (interesting) contents of ``s``, leaving the shrinker stuck on a
larger example than necessary; we now realign by truncating the recorded value,
preserving content from either end.  This also helps :ref:`stateful tests
<stateful>` whose rules draw such size-dependent values (:issue:`4006`).
