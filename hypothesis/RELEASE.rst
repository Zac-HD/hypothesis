RELEASE_TYPE: minor

|st.times| and |st.datetimes| now rewrite simple comparison filters such as
``.filter(partial(operator.ge, bound))``, as |st.dates| already did.

For naive times and datetimes we narrow ``min_value`` and ``max_value``
directly, so contradictory filters give an empty strategy instead of failing
health checks.  For aware datetimes, filtering by an aware bound now narrows
each draw to (approximately) the feasible window once the timezone is known,
retaining the predicate so that results are exact even near a DST transition.
This also makes strategies inferred for types like
``Annotated[datetime, annotated_types.Gt(bound)]`` far more efficient.
