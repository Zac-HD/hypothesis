RELEASE_TYPE: patch

This release improves |Phase.explain| reports (:issue:`3551`):

* report up to ten suspicious lines, instead of five
* skip lines which only ran after the failing exception was raised, since
  they cannot have caused the failure
* when reporting several lines, mark the one executed earliest as the
  ``(earliest divergence)``, as the most likely place to start debugging
