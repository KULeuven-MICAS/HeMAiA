# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Make the grouped ``util/sim`` subdirs importable with flat module names.

The shared sim helpers were regrouped into ``common/``, ``gemm/``, ``xdma/`` and
``ara/`` subpackages, but most consumers still ``from data_utils import ...``.
Importing this module (after ``util/sim`` is on ``sys.path``) appends those four
subdirs to ``sys.path`` so the flat imports keep resolving. This replaces a
7-line bootstrap that used to be copy-pasted into every datagen/main_bingo.
"""

import os
import sys

for _p in [p for p in list(sys.path) if str(p).rstrip("/").endswith("util/sim")]:
    for _s in ("common", "gemm", "xdma", "ara"):
        _sub = os.path.join(_p, _s)
        if _sub not in sys.path:
            sys.path.append(_sub)
