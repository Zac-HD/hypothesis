# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import re
import tomllib
from pathlib import Path

import pytest

from hypothesistooling.__main__ import EXTRAS, check_requirements

PYPROJECT = tomllib.loads(Path("hypothesis/pyproject.toml").read_text(encoding="utf-8"))
EXTRA_DEPS = PYPROJECT["project"]["optional-dependencies"]
GROUPS = tomllib.loads(
    Path("hypothesis/ci-matrix/pyproject.toml").read_text(encoding="utf-8")
)["dependency-groups"]
TOX_INI = Path("hypothesis/tox.ini").read_text(encoding="utf-8")


def test_requirements():
    check_requirements()


@pytest.mark.parametrize(
    "group, extra, package",
    [("oldestnumpy", "numpy", "numpy"), ("larkmin", "lark", "lark")],
)
def test_minimum_version_groups_match_our_declared_floor(group, extra, package):
    # These groups exist to test the oldest version we claim to support, so they
    # have to track the floor in [project.optional-dependencies] by hand.
    (floor,) = re.findall(rf"{package}>=([0-9.]+)", str(EXTRA_DEPS[extra]))
    (pinned,) = re.findall(rf"{package}==([0-9.]+)", str(GROUPS[group]))
    assert floor == pinned, f"{group} should pin {package}=={floor}"


def test_every_extra_environment_is_registered():
    (factors,) = re.findall(r"\[testenv:extra-\{([a-z,]+)\}\]", TOX_INI)
    assert set(factors.split(",")) == set(EXTRAS)


def test_nothing_is_installed_at_test_time():
    # The whole point of #4494: if a test environment installs something, that
    # something is not in uv.lock, and upstream can break CI without us doing
    # anything.  numpy-nightly is the sole intentional exception - it exists to
    # fail when a future numpy breaks us.
    nightly = TOX_INI.index("[testenv:numpy-nightly]")
    end = TOX_INI.index("[testenv:", nightly + 1)
    remainder = TOX_INI[:nightly] + TOX_INI[end:]
    assert "pip install" not in remainder
