# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Reaching util/sim from inside libs.

The root itself is found by `_bingo_paths.repo_root`, which searches for a marker instead
of counting `..`. There is deliberately only one implementation of that search: two would
drift, and the way this fails is silent -- a wrong root is simply a directory that does not
exist, so the sys.path entry is useless and the error surfaces as a ModuleNotFoundError
from a module that never mentions paths.
"""

import os
import sys

import _bingo_paths


def repo_root(start: str = None) -> str:
    return _bingo_paths.repo_root()


def add_sim_paths(*subdirs) -> str:
    """Put util/sim/<subdir> on sys.path for each named subdir. Returns the root."""
    root = repo_root()
    for s in subdirs:
        p = os.path.join(root, "util", "sim", s)
        if p not in sys.path:
            sys.path.append(p)
    return root
