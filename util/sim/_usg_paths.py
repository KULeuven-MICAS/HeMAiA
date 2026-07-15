# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Make the grouped ``util/sim`` subdirs importable with flat module names.

The shared sim helpers live in the ``common/``, ``gemm/``, ``xdma/`` and ``ara/``
subdirs, but consumers import them flat (``from data_utils import ...``).
Importing this module -- once ``util/sim`` itself is on ``sys.path`` -- appends
those four subdirs to ``sys.path``, which is what makes the flat imports resolve.
Every datagen / main_bingo that uses the helpers must import it first.
"""

import os
import sys

for _p in [p for p in list(sys.path) if str(p).rstrip("/").endswith("util/sim")]:
    for _s in ("common", "gemm", "xdma", "ara"):
        _sub = os.path.join(_p, _s)
        if _sub not in sys.path:
            sys.path.append(_sub)
