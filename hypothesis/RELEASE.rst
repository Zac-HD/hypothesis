RELEASE_TYPE: patch

This release improves |Phase.explain| reports (:issue:`3551`):

* skip lines which only ran after the failing exception was raised, since
  they cannot have caused the failure
* print suspicious lines in the order they first executed, so that the
  earliest divergence comes first
* show up to ten lines, up from five, truncating the middle of longer
  reports rather than the tail
