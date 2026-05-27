RELEASE_TYPE: patch

This patch makes |fuzz_one_input| work on instance methods, e.g.
``instance.test_method.hypothesis.fuzz_one_input(...)``, by passing the bound
instance through as the test's ``self`` argument.  Previously this raised a
``TypeError`` about a missing ``self`` argument (:issue:`4060`).
