RELEASE_TYPE: patch

This release improves |Phase.explain| reports (:issue:`3551`):

* skip lines which only ran after the failing exception was raised, since
  they cannot have caused the failure
* print suspicious lines in the order they first executed, so that the
  earliest divergence comes first
* show up to ten lines, up from five, preferring to drop stdlib and then
  site-packages locations over local code when truncating
