RELEASE_TYPE: minor

This release adds the |PrimitiveProvider.observability| attribute, which lets
:ref:`alternative backends <alternative-backends>` declare what
:ref:`observability <observability>` data they need - such as choice sequences
without coverage information - instead of the previous all-or-nothing
``add_observability_callback`` flag, which is now deprecated.

Backends now also no longer enable coverage collection as a side effect of
opting into observations.
