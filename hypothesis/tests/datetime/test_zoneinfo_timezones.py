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
import operator
import platform
import zoneinfo
from functools import partial

import pytest

from hypothesis import given, settings, strategies as st
from hypothesis.errors import InvalidArgument

from tests.common.debug import assert_no_examples, find_any, minimal


def test_utc_is_minimal():
    assert minimal(st.timezones()) is zoneinfo.ZoneInfo("UTC")


def test_can_generate_non_utc():
    find_any(
        st.datetimes(timezones=st.timezones()).filter(lambda d: d.tzinfo.key != "UTC")
    )


@settings(max_examples=25)
@given(
    st.datetimes(timezones=st.timezones())
    .filter(partial(operator.le, dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc)))
    .filter(partial(operator.ge, dt.datetime(2020, 4, 1, tzinfo=dt.timezone.utc)))
)
def test_aware_datetimes_filter_rewriting_narrows_arbitrary_timezones(value):
    # This spans a DST transition in many timezones; the rewritten strategy
    # narrows each draw to near the three-month window and the retained
    # predicate ensures exactness.
    assert dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc) <= value
    assert value <= dt.datetime(2020, 4, 1, tzinfo=dt.timezone.utc)


@given(st.data(), st.datetimes(), st.datetimes())
def test_datetimes_stay_within_naive_bounds(data, lo, hi):
    if lo > hi:
        lo, hi = hi, lo
    out = data.draw(st.datetimes(lo, hi, timezones=st.timezones()))
    assert lo <= out.replace(tzinfo=None) <= hi


@pytest.mark.parametrize("kwargs", [{"no_cache": 1}])
def test_timezones_argument_validation(kwargs):
    with pytest.raises(InvalidArgument):
        st.timezones(**kwargs).validate()


@pytest.mark.parametrize(
    "kwargs",
    [
        # {"allow_alias": 1},
        # {"allow_deprecated": 1},
        {"allow_prefix": 1},
    ],
)
def test_timezone_keys_argument_validation(kwargs):
    with pytest.raises(InvalidArgument):
        st.timezone_keys(**kwargs).validate()


@pytest.mark.xfail(strict=False, reason="newly failing on GitHub Actions")
@pytest.mark.skipif(platform.system() != "Linux", reason="platform-specific")
def test_can_generate_prefixes_if_allowed_and_available():
    """
    This is actually kinda fiddly: we may generate timezone keys with the
    "posix/" or "right/" prefix if-and-only-if they are present on the filesystem.

    This immediately rules out Windows (which uses the tzdata package instead),
    along with OSX (which doesn't seem to have prefixed keys).  We believe that
    they are present on at least most Linux distros, but have not done exhaustive
    testing.

    It's fine to just patch this test out if it fails - passing in the
    Hypothesis CI demonstrates that the feature works on *some* systems.
    """
    find_any(st.timezone_keys(), lambda s: s.startswith("posix/"))
    find_any(st.timezone_keys(), lambda s: s.startswith("right/"))


def test_can_disallow_prefixes():
    assert_no_examples(
        st.timezone_keys(allow_prefix=False),
        lambda s: s.startswith(("posix/", "right/")),
    )
