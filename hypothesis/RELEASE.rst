RELEASE_TYPE: patch

This patch fixes a race condition where computing the internal ``is_empty``,
``has_reusable_values``, or ``is_cacheable`` properties of a strategy from
multiple threads at once could raise an :class:`AttributeError` (:issue:`4475`).
This was only possible on the free-threaded builds of CPython.
