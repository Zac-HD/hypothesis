# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import datetime as dt
from calendar import monthrange

import pytest

from hypothesis import HealthCheck, example, given, settings, strategies as st
from hypothesis.errors import InvalidArgument
from hypothesis.strategies import dates, datetimes, timedeltas, times
from hypothesis.strategies._internal.datetime import _num_days_in_month

from tests.common.debug import assert_simple_property, find_any, minimal
from tests.common.utils import fails_with


@example(1900)
@example(2000)
@example(2004)
@example(2100)
@given(st.integers(dt.MINYEAR, dt.MAXYEAR))
def test_num_days_in_month_matches_monthrange(year):
    for month in range(1, 13):
        assert _num_days_in_month(year, month) == monthrange(year, month)[1]


def test_can_find_positive_delta():
    assert minimal(timedeltas(), lambda x: x.days > 0) == dt.timedelta(1)


def test_can_find_negative_delta():
    assert minimal(
        timedeltas(max_value=dt.timedelta(10**6)), lambda x: x.days < 0
    ) == dt.timedelta(-1)


def test_can_find_on_the_second():
    find_any(timedeltas(), lambda x: x.seconds == 0)


def test_can_find_off_the_second():
    find_any(timedeltas(), lambda x: x.seconds != 0)


def test_simplifies_towards_zero_delta():
    d = minimal(timedeltas())
    assert d.days == d.seconds == d.microseconds == 0


def test_min_value_is_respected():
    assert minimal(timedeltas(min_value=dt.timedelta(days=10))).days == 10


def test_max_value_is_respected():
    assert minimal(timedeltas(max_value=dt.timedelta(days=-10))).days == -10


@settings(suppress_health_check=list(HealthCheck))
@given(timedeltas())
def test_single_timedelta(val):
    assert_simple_property(timedeltas(val, val), lambda v: v is val)


def test_simplifies_towards_millenium():
    d = minimal(datetimes())
    assert d.year == 2000
    assert d.month == d.day == 1
    assert d.hour == d.minute == d.second == d.microsecond == 0


@given(datetimes())
def test_default_datetimes_are_naive(dt):
    assert dt.tzinfo is None


def test_bordering_on_a_leap_year():
    x = minimal(
        datetimes(
            dt.datetime.min.replace(year=2003), dt.datetime.max.replace(year=2005)
        ),
        lambda x: x.month == 2 and x.day == 29,
        settings=settings(max_examples=2500),
    )
    assert x.year == 2004


def test_can_find_after_the_year_2000():
    assert minimal(dates(), lambda x: x.year > 2000).year == 2001


def test_can_find_before_the_year_2000():
    assert minimal(dates(), lambda x: x.year < 2000).year == 1999


@pytest.mark.parametrize("month", range(1, 13))
def test_can_find_each_month(month):
    find_any(dates(), lambda x: x.month == month, settings(max_examples=10**6))


def test_min_year_is_respected():
    assert minimal(dates(min_value=dt.date.min.replace(2003))).year == 2003


def test_max_year_is_respected():
    assert minimal(dates(max_value=dt.date.min.replace(1998))).year == 1998


@given(dates())
def test_single_date(val):
    assert find_any(dates(val, val)) is val


def test_can_find_midnight():
    find_any(times(), lambda x: x.hour == x.minute == x.second == 0)


def test_can_find_non_midnight():
    assert minimal(times(), lambda x: x.hour != 0).hour == 1


def test_can_find_on_the_minute():
    find_any(times(), lambda x: x.second == 0)


def test_can_find_off_the_minute():
    find_any(times(), lambda x: x.second != 0)


def test_simplifies_towards_midnight():
    d = minimal(times())
    assert d.hour == d.minute == d.second == d.microsecond == 0


def test_can_generate_naive_time():
    find_any(times(), lambda d: not d.tzinfo)


@given(times())
def test_naive_times_are_naive(dt):
    assert dt.tzinfo is None


def test_can_generate_datetime_with_fold_1():
    find_any(datetimes(), lambda d: d.fold)


def test_can_generate_time_with_fold_1():
    find_any(times(), lambda d: d.fold)


@given(datetimes(allow_imaginary=False))
def test_allow_imaginary_is_not_an_error_for_naive_datetimes(d):
    pass


UTC_2000 = dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc)
PLUS_FIVE_THIRTY = dt.timezone(dt.timedelta(hours=5, minutes=30))


@given(datetimes(min_value=UTC_2000, max_value=UTC_2000 + dt.timedelta(days=1)))
def test_aware_bounds_default_timezones_to_the_bounds_tzinfo(value):
    # Aware bounds constrain the instant of generated datetimes, and default
    # the timezones strategy to the timezone(s) of the bounds.
    assert UTC_2000 <= value <= UTC_2000 + dt.timedelta(days=1)
    assert value.tzinfo is dt.timezone.utc


@given(
    datetimes(
        min_value=UTC_2000,
        max_value=dt.datetime(2000, 6, 1, tzinfo=PLUS_FIVE_THIRTY),
    )
)
def test_aware_bounds_infer_timezones_from_both_bounds(value):
    assert UTC_2000 <= value <= dt.datetime(2000, 6, 1, tzinfo=PLUS_FIVE_THIRTY)
    assert value.tzinfo in (dt.timezone.utc, PLUS_FIVE_THIRTY)


def test_aware_bounds_can_generate_all_inferred_timezones():
    strategy = datetimes(
        min_value=UTC_2000, max_value=dt.datetime(2000, 6, 1, tzinfo=PLUS_FIVE_THIRTY)
    )
    find_any(strategy, lambda d: d.tzinfo is dt.timezone.utc)
    find_any(strategy, lambda d: d.tzinfo is PLUS_FIVE_THIRTY)


@given(datetimes(min_value=UTC_2000, max_value=UTC_2000))
def test_aware_bounds_can_pin_a_single_instant(value):
    assert value == UTC_2000


@given(
    datetimes(
        max_value=dt.datetime.max.replace(tzinfo=dt.timezone.utc)
        - dt.timedelta(hours=12),
        timezones=st.just(dt.timezone(dt.timedelta(hours=14))),
    )
)
def test_aware_bound_near_end_of_representable_range(value):
    # The bound has no wall-clock representation in UTC+14, which is fine.
    assert value <= dt.datetime.max.replace(tzinfo=dt.timezone.utc) - dt.timedelta(
        hours=12
    )


@given(
    datetimes(
        min_value=dt.datetime.min.replace(tzinfo=dt.timezone.utc)
        + dt.timedelta(hours=12),
        timezones=st.just(dt.timezone(dt.timedelta(hours=-14))),
    )
)
def test_aware_bound_near_start_of_representable_range(value):
    assert value >= dt.datetime.min.replace(tzinfo=dt.timezone.utc) + dt.timedelta(
        hours=12
    )


@given(
    datetimes(
        min_value=dt.datetime.max.replace(tzinfo=dt.timezone.utc)
        - dt.timedelta(hours=12),
        timezones=st.sampled_from(
            [dt.timezone.utc, dt.timezone(dt.timedelta(hours=14))]
        ),
    )
)
def test_aware_min_bound_may_be_unsatisfiable_in_some_timezones(value):
    # No wall-clock time in UTC+14 has an instant this late, so we only ever
    # generate UTC datetimes here.
    assert value.tzinfo is dt.timezone.utc
    assert value >= dt.datetime.max.replace(tzinfo=dt.timezone.utc) - dt.timedelta(
        hours=12
    )


@given(
    datetimes(
        max_value=dt.datetime.min.replace(tzinfo=dt.timezone.utc)
        + dt.timedelta(hours=12),
        timezones=st.sampled_from(
            [dt.timezone.utc, dt.timezone(dt.timedelta(hours=-14))]
        ),
    )
)
def test_aware_max_bound_may_be_unsatisfiable_in_some_timezones(value):
    assert value.tzinfo is dt.timezone.utc
    assert value <= dt.datetime.min.replace(tzinfo=dt.timezone.utc) + dt.timedelta(
        hours=12
    )


@fails_with(InvalidArgument)
@given(
    datetimes(min_value=UTC_2000, timezones=st.sampled_from([None, dt.timezone.utc]))
)
def test_aware_bounds_reject_a_none_timezone_at_draw_time(value):
    pass


class NoOffsetTimezone(dt.tzinfo):
    def utcoffset(self, value):
        return None

    def tzname(self, value):
        return "?"

    def dst(self, value):
        return None


def test_aware_bound_without_utc_offset_is_invalid():
    with pytest.raises(InvalidArgument, match="no UTC offset"):
        datetimes(
            min_value=dt.datetime(2000, 1, 1, tzinfo=NoOffsetTimezone())
        ).validate()
