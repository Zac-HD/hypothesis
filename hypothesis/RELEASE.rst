RELEASE_TYPE: minor

This release stabilizes :ref:`observability <observability>`, which is no
longer experimental.

When observability is enabled, Hypothesis records data about each test case -
including a timing breakdown, coverage information, a representation of the
arguments, and the outcome - and by default writes it to the
``.hypothesis/observed`` directory in an analysis-ready
`jsonlines <https://jsonlines.org/>`_ format.

Observability can now be enabled with the new |settings.observability| setting,
and configured with the new |ObservabilityConfig| class - including in-memory
delivery to your own callbacks, and opt-in recording of the choice sequence.
See :ref:`Configuring observability <observability-configuration>` for details.

The experimental ``HYPOTHESIS_EXPERIMENTAL_OBSERVABILITY`` and
``HYPOTHESIS_EXPERIMENTAL_OBSERVABILITY_CHOICES`` environment variables are
deprecated in favor of |settings.observability| - to configure observability
from your environment, use a settings profile conditioned on an environment
variable of your choice.  ``HYPOTHESIS_EXPERIMENTAL_OBSERVABILITY_NOCOVER``
has been removed - pass ``ObservabilityConfig(coverage=False)`` instead.
