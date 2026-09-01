# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import os
import pickle
import uuid

import pytest

from hypothesis import settings

from tests.common.utils import time_sleep
from tests.cover.test_database_backend import _database_conforms_to_listener_api
from tests.cover.test_database_structured import (
    ForwardingDatabase,
    conforms_to_structured_api,
    journal_records_entries_deleted_by_id,
)

pytest.importorskip("psycopg")

from hypothesis.extra.postgres import PostgresExampleDatabase

# Set this to a libpq connection string to run these tests.
POSTGRES_URL = os.environ.get("HYPOTHESIS_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    POSTGRES_URL is None, reason="needs HYPOTHESIS_TEST_POSTGRES_URL"
)


@pytest.fixture(autouse=True)
def _consistently_increment_time():
    """Use the real clock, not the test suite's fake one, because these tests
    wait for other threads and processes."""


def _database(_path=None):
    return PostgresExampleDatabase(POSTGRES_URL, namespace=uuid.uuid4().hex)


def test_structured_api():
    conforms_to_structured_api(
        _database, parent_settings=settings(max_examples=50, stateful_step_count=30)
    )


def test_listener_api():
    _database_conforms_to_listener_api(
        _database,
        flush=None,
        parent_settings=settings(max_examples=5, stateful_step_count=10, deadline=None),
    )


@pytest.fixture
def make_db():
    made = []

    def make():
        made.append(_database())
        return made[-1]

    yield make
    for db in made:
        db.close()


def test_map_entries_expire(make_db):
    db = make_db()
    db.map_put(("p", "m"), "short", b"1", ttl=0.5)
    db.map_put(("p", "m"), "long", b"2", ttl=100)
    time_sleep(1)
    assert db.map_items(("p", "m")) == {("long",): b"2"}
    assert db.map_put(("p", "m"), "short", b"3", expect=None)


@pytest.mark.parametrize("wrap", [lambda db: db, ForwardingDatabase])
def test_journal_records_entries_deleted_by_id(make_db, wrap):
    journal_records_entries_deleted_by_id(wrap(make_db()))


def test_namespaces_are_separate(make_db):
    a, b = make_db(), make_db()
    a.save(b"key", b"value")
    assert list(b.fetch(b"key")) == []
    assert list(a.fetch(b"key")) == [b"value"]


def test_can_be_pickled(make_db):
    db = make_db()
    db.save(b"key", b"value")
    copy = pickle.loads(pickle.dumps(db))
    try:
        assert list(copy.fetch(b"key")) == [b"value"]
    finally:
        copy.close()
