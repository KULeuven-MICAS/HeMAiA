# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The decode's checks: every array task a GEMV, Y in HBM and the root's A8 row."""

import numpy as np

import gemv_utils


class DChecks:
    """DBuild._checks, a8_root and the GEMV-only check of the graph."""

    def _check_gemv_only(self, first):
        """Every array task this block made (node ids from `first`) is a GEMV, so the GEMM-config reuse (which tracks
        only GEMM kernels) finds no GEMM here to turn into a gemm_minimal behind a GEMV."""
        gemm = self.roles["gemm"]
        arr = [t for t in self.dfg.node_list if t.node_id >= first and t.assigned_core_id == gemm
               and (t.kernel_name or "").startswith("__snax")]
        bad = [t.node_name for t in arr if t.kernel_name != "__snax_bingo_kernel_gemv"]
        if bad:
            raise ValueError(f"array tasks that are not GEMVs: {bad[:4]}")
        print(f"[{self.TAG}] {len(arr)} array tasks, every one a GEMV (gate|up and down)")

    def a8_root(self):
        """(the root's cluster-0 A8 buffer, its golden bytes, the bytes checked): the chip's A8 row in a_row (its free
        bytes zero), up to 2 KiB. A layer's block checks the same."""
        R = self.ROOT
        cl = self.clusters(R)[0]
        c0, ncol = self.kcols[R]                # the chip's A8, both clusters
        img = gemv_utils.a_row_bytes(self.d.a8[0, c0:c0 + ncol]).view(np.uint8)
        n = min(2048, img.size)
        return self.a8[cl][0], img[:n], n

    def _checks(self):
        """Y_raw's row (y_dequant: Y16) in the HBM of the root's memory chip, every column (FP16, column order); the
        root's own A8 (a8_root)."""
        R = self.ROOT
        y = np.ascontiguousarray(self.gold["y16" if self.y_dequant else "y_raw"][0].astype(np.float16)).view(np.uint8)
        checks = [("y_hbm", self.h_yhbm, y, y.size)]
        buf, img, n = self.a8_root()
        got = self.stash("a8_root", R, 0, buf, n, [])
        checks.append(("a8_root", got, img, n))
        self.run_checks(checks)
