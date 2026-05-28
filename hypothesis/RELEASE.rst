RELEASE_TYPE: patch

This patch makes |fuzz_one_input| work on instance methods, e.g.
``instance.test_method.hypothesis.fuzz_one_input(...)``, by capturing ``self``
from the most recent normal call to the method.  The method must therefore
have been called at least once before being fuzzed (:issue:`4060`).
