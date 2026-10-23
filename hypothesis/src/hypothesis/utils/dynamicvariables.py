# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import threading
from types import TracebackType
from typing import Generic, TypeVar

T = TypeVar("T")


class DynamicVariable(Generic[T]):
    def __init__(self, default: T) -> None:
        self.default = default
        self.data = threading.local()

    @property
    def value(self) -> T:
        return getattr(self.data, "value", self.default)

    @value.setter
    def value(self, value: T) -> None:
        self.data.value = value

    def with_value(self, value: T) -> "_WithValue[T]":
        return _WithValue(self, value)


class _WithValue(Generic[T]):
    # A class rather than @contextmanager, because this is on the hot path for
    # every test case and generator-based context managers are much slower.
    __slots__ = ("_old_value", "_value", "_variable")

    def __init__(self, variable: DynamicVariable[T], value: T) -> None:
        self._variable = variable
        self._value = value

    def __enter__(self) -> None:
        self._old_value = self._variable.value
        self._variable.data.value = self._value

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._variable.data.value = self._old_value
        # Like a generator-based context manager, we're single-use, and dropping
        # these references avoids reference cycles (e.g. via BuildContext).
        del self._old_value, self._value
