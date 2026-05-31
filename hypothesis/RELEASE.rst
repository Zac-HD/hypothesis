RELEASE_TYPE: minor

This release adds support for the |.map| and |.filter| methods on
:class:`~hypothesis.stateful.Bundle`, by reusing the internals of
:func:`~hypothesis.strategies.sampled_from`.  You can now write rules such as
``@rule(value=consumes(my_bundle).filter(...))`` or
``@rule(value=my_bundle.map(...))``, and filtering retries within a single draw
rather than rejecting the whole step - so it no longer consumes more elements
than intended from a ``consumes()`` bundle (:issue:`3944`).
