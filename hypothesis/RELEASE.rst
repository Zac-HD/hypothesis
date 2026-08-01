RELEASE_TYPE: patch

This patch refactors :ref:`observability <observability>` internals, in
preparation for stabilizing observability (:issue:`4387`). As part of this,
Hypothesis no longer touches the ``.hypothesis/observed`` directory at import
time; week-old observation files are now removed when a test runs instead.
