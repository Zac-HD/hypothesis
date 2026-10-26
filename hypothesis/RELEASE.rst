RELEASE_TYPE: patch

This patch adds a ``# pragma: no cover`` comment to the code path for
:pypi:`dateutil` versions before 3.0, which our coverage tests no longer use.
