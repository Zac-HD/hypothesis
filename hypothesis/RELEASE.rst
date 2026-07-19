RELEASE_TYPE: patch

On Python 3.12 and later, our :mod:`sys.monitoring` integration (used by
|Phase.explain| and :ref:`observability <observability>` coverage) now traces
branches instead of lines (:issue:`3781`). This substantially reduces tracing
overhead, especially on Python 3.14+ where each branch is traced at most once
per test case.
