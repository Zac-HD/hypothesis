RELEASE_TYPE: minor

:func:`~hypothesis.strategies.fixed_dictionaries` now generates ``dict``\ s
with varied iteration orders, rather than always placing the required keys
first (in the order they were passed) followed by the optional keys.  Since
``dict`` iteration order has been observable since Python 3.7, this helps to
find bugs in code which is sensitive to key order (:issue:`3906`).

The set of keys - and therefore the distribution of dictionary sizes - is
unchanged, and examples still shrink towards the original key order.  If you
need a specific key order, construct it explicitly, e.g. with
``st.tuples(...).map(...)``.
