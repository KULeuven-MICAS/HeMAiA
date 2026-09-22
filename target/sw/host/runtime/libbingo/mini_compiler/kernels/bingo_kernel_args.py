# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Every kernel's argument struct, in one importable name.

This file used to BE all of them -- 241 classes in 3,382 lines, which is not a module so
much as a filing cabinet. The classes now live one file per engine:

    kernel_base.py      the BingoKernelArgs ABC: address resolution, C field emission,
                        and the placement-order constraint a few kernels declare
    kernel_misc.py      dummy / sync probe / sync report, and the cluster iDMA
    kernel_gemm.py      VersaCore GEMM, general and typed
    kernel_gemm_fa.py   the FlashAttention QK/PV pair and the array's counters
    kernel_xdma.py      xDMA transfers, 2-D shape ops, and the in-fabric junction CSRs
    kernel_layout.py    the dedicated layout converters (D<->row, row<->A, row<->B)
    kernel_simd.py      SIMD streaming primitives and the fused whole-operators
    kernel_host.py      host transfers and the per-precision result checks
    kernel_ara.py       Ara (RVV) host kernels, typed by precision

WHY THIS FILE REMAINS. Roughly sixty workloads import from `bingo_kernel_args` by name, and
splitting the file is not worth changing every one of them. So this is a facade: it
re-exports the lot, the flat name keeps working, and new code can import the narrower
module when it only wants one engine.

Adding a kernel means adding its class to the engine's file. Nothing here needs editing --
the star imports pick it up -- unless it is a new engine, which needs a line below.
"""

from kernel_base import *        # noqa: F401,F403
from kernel_base import BingoKernelArgs  # noqa: F401  (explicit: the ABC everything subclasses)
from kernel_misc import *        # noqa: F401,F403
from kernel_gemm import *        # noqa: F401,F403
from kernel_gemm_fa import *     # noqa: F401,F403
from kernel_xdma import *        # noqa: F401,F403
from kernel_layout import *      # noqa: F401,F403
from kernel_simd import *        # noqa: F401,F403
from kernel_host import *        # noqa: F401,F403
from kernel_ara import *         # noqa: F401,F403

# Private names a star import will not carry, but that callers inside the compiler use.
from kernel_host import _CHECK_TYPE_ELEM_BYTES  # noqa: F401
