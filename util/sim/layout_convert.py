"""
VersaCore blocked-layout conversions.

Defines 6 primitive conversions between row-major (logical 2D) and the
three VersaCore blocked layouts. All conversions are pure index permutations
— no data transformation, just reordering of elements.

Layouts:
    - Row-major:    R has shape (rows, cols), flat index = i * cols + j
    - A-layout:     [M_T, K_T, meshRow, tileSize] flat
                    A_stored[m, k, r, s] = A_logical[m*meshRow + r, k*tileSize + s]
    - B-layout:     [N_T, K_T, meshCol, tileSize] flat
                    B_stored[n, k, c, s] = B_logical[k*tileSize + s, n*meshCol + c]
    - D-layout:     [M_T, N_T, meshRow, meshCol] flat
                    D_stored[m, n, r, c] = D_logical[m*meshRow + r, n*meshCol + c]

These functions are used by:
    1. Hand-written attention_datagen.py (via a reusable import)
    2. Emitted datagen.py files (from codegen/onnx_to_hemaia/)
    3. Golden generators for xdma_layout_transform/ workloads
"""

from __future__ import annotations

import numpy as np


# ── Row-major → blocked layouts ────────────────────────────────────────


def row_major_to_a(R: np.ndarray, M_T: int, K_T: int,
                    meshRow: int, tileSize: int) -> np.ndarray:
    """Convert row-major (M_T*meshRow, K_T*tileSize) → A-layout flat.

    A_stored[m, k, r, s] = R[m*meshRow + r, k*tileSize + s]
    """
    assert R.shape == (M_T * meshRow, K_T * tileSize), \
        f"expected shape ({M_T*meshRow}, {K_T*tileSize}), got {R.shape}"
    return (R.reshape(M_T, meshRow, K_T, tileSize)
             .transpose(0, 2, 1, 3)
             .reshape(-1))


def row_major_to_b(R: np.ndarray, K_T: int, N_T: int,
                    tileSize: int, meshCol: int) -> np.ndarray:
    """Convert row-major (K_T*tileSize, N_T*meshCol) → B-layout flat.

    B_stored[n, k, c, s] = R[k*tileSize + s, n*meshCol + c]
    """
    assert R.shape == (K_T * tileSize, N_T * meshCol), \
        f"expected shape ({K_T*tileSize}, {N_T*meshCol}), got {R.shape}"
    return (R.reshape(K_T, tileSize, N_T, meshCol)
             .transpose(2, 0, 3, 1)
             .reshape(-1))


def row_major_to_d(R: np.ndarray, M_T: int, N_T: int,
                    meshRow: int, meshCol: int) -> np.ndarray:
    """Convert row-major (M_T*meshRow, N_T*meshCol) → D-layout flat.

    D_stored[m, n, r, c] = R[m*meshRow + r, n*meshCol + c]
    """
    assert R.shape == (M_T * meshRow, N_T * meshCol), \
        f"expected shape ({M_T*meshRow}, {N_T*meshCol}), got {R.shape}"
    return (R.reshape(M_T, meshRow, N_T, meshCol)
             .transpose(0, 2, 1, 3)
             .reshape(-1))


# ── Blocked layouts → row-major (inverses) ─────────────────────────────


def a_to_row_major(A_flat: np.ndarray, M_T: int, K_T: int,
                    meshRow: int, tileSize: int) -> np.ndarray:
    """Inverse of row_major_to_a. Returns (M_T*meshRow, K_T*tileSize) array."""
    expected = M_T * K_T * meshRow * tileSize
    assert A_flat.size == expected, \
        f"expected {expected} elements, got {A_flat.size}"
    return (A_flat.reshape(M_T, K_T, meshRow, tileSize)
                  .transpose(0, 2, 1, 3)
                  .reshape(M_T * meshRow, K_T * tileSize))


def b_to_row_major(B_flat: np.ndarray, K_T: int, N_T: int,
                    tileSize: int, meshCol: int) -> np.ndarray:
    """Inverse of row_major_to_b. Returns (K_T*tileSize, N_T*meshCol) array."""
    expected = N_T * K_T * meshCol * tileSize
    assert B_flat.size == expected, \
        f"expected {expected} elements, got {B_flat.size}"
    return (B_flat.reshape(N_T, K_T, meshCol, tileSize)
                  .transpose(1, 3, 0, 2)
                  .reshape(K_T * tileSize, N_T * meshCol))


def d_to_row_major(D_flat: np.ndarray, M_T: int, N_T: int,
                    meshRow: int, meshCol: int) -> np.ndarray:
    """Inverse of row_major_to_d. Returns (M_T*meshRow, N_T*meshCol) array."""
    expected = M_T * N_T * meshRow * meshCol
    assert D_flat.size == expected, \
        f"expected {expected} elements, got {D_flat.size}"
    return (D_flat.reshape(M_T, N_T, meshRow, meshCol)
                  .transpose(0, 2, 1, 3)
                  .reshape(M_T * meshRow, N_T * meshCol))


# ── Explicit index formulas (used by C kernel implementations) ─────────


def a_offset(m: int, k: int, r: int, s: int,
             M_T: int, K_T: int, meshRow: int, tileSize: int) -> int:
    """Byte offset of A_stored[m, k, r, s] in flat A-layout."""
    return ((m * K_T + k) * meshRow + r) * tileSize + s


def b_offset(n: int, k: int, c: int, s: int,
             N_T: int, K_T: int, meshCol: int, tileSize: int) -> int:
    """Byte offset of B_stored[n, k, c, s] in flat B-layout."""
    return ((n * K_T + k) * meshCol + c) * tileSize + s


def d_offset(m: int, n: int, r: int, c: int,
             M_T: int, N_T: int, meshRow: int, meshCol: int) -> int:
    """Byte offset of D_stored[m, n, r, c] in flat D-layout."""
    return ((m * N_T + n) * meshRow + r) * meshCol + c


def rm_offset(i: int, j: int, cols: int) -> int:
    """Byte offset of R[i, j] in flat row-major."""
    return i * cols + j
