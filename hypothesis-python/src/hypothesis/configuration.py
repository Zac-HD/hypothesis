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
from pathlib import Path

__hypothesis_home_directory_default = Path.cwd() / ".hypothesis"

__hypothesis_home_directory = None


def set_hypothesis_home_dir(directory: os.PathLike) -> None:
    global __hypothesis_home_directory
    __hypothesis_home_directory = None if directory is None else Path(directory)


def check_not_loading_plugins(what: str, /) -> None:
    """Avoid touching disk or materializing lazy/deferred strategies from plugins.

    Currently a deprecation warning; this will become an error in future.
    """
    # Indirection so that we can disable it once plugin loading is done.
    __check_not_loading_plugins_inner(what)


def __check_not_loading_plugins_inner(what: str, /) -> None:  # pragma: no cover
    import hypothesis
    from hypothesis._settings import note_deprecation

    if not hasattr(hypothesis, "run"):
        global __check_not_loading_plugins_inner
        __check_not_loading_plugins_inner = lambda _: None
        return

    note_deprecation(
        f"Slow code in plugin: avoid {what} at import time!  This will be an error "
        "in a future version of Hypothesis; use -Werror to get a traceback and show "
        "which plugin is responsible.",
        since="RELEASEDAY",
        has_codemod=False,
        stacklevel=3,
    )


def storage_directory(*names: str) -> Path:
    check_not_loading_plugins(f"accessing storage for {'/'.join(names)}")
    global __hypothesis_home_directory
    if not __hypothesis_home_directory:
        if where := os.getenv("HYPOTHESIS_STORAGE_DIRECTORY"):
            __hypothesis_home_directory = Path(where)
    if not __hypothesis_home_directory:
        __hypothesis_home_directory = __hypothesis_home_directory_default
    return __hypothesis_home_directory.joinpath(*names)
