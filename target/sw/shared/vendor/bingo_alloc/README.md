# Bingo Allocator

A deterministic heap allocator for HeMAiA's heterogeneous SoC, derived from
[o1heap](https://github.com/pavel-kirienko/o1heap) (MIT licensed, Pavel Kirienko).

## Why not o1heap directly?

o1heap rounds every allocation request up to the **next power of 2**. On a
512 KB TCDM scratchpad this limits the maximum single allocation to ~256 KB:

```
o1heapAllocate(handle, 300 * 1024)
  -> fragment_size = roundUpToPow2(300 KB + 128) = 512 KB
  -> 512 KB > usable capacity (~508 KB)  -> returns NULL
```

For the DNN workloads, the capacity is more important than the predictive allocation latency.

The bingo allocator fixes this.

## What changed

| Aspect | o1heap | bingo_alloc |
|--------|--------|-------------|
| Request rounding | Next power of 2 | Next multiple of 256 bytes (FRAGMENT_SIZE_MIN) |
| Max single alloc (512 KB TCDM) | ~256 KB | ~508 KB |
| Bin lookup | O(1) guaranteed (first fragment always fits) | O(1) typical; O(k) in the optimal bin when fragments vary in size |
| Coalescing / free | Identical | Identical |
| 32/64-bit shared memory | Identical (`__riscv_xlen` dispatch) | Identical |

### Allocation algorithm

1. Compute `fragment_size = align_up(amount + 128, 256)` (not `roundUpToPow2`).
2. Find the optimal bin via `log2Floor(fragment_size / FRAGMENT_SIZE_MIN)`.
3. If the first fragment in that bin is large enough, use it (O(1) path).
4. Otherwise, walk the bin's free list looking for a fit.
5. If no fragment in the optimal bin fits, fall back to the next higher bin
   where every fragment is guaranteed to be large enough (O(1)).

In practice, TCDM heaps hold very few fragments (2-10 tensors), so the
within-bin search is effectively constant time.

## Heterogeneous SoC support

The allocator stores all pointers as `uint64_t` values rather than native
pointers. Casting macros check `__riscv_xlen` at compile time:

- **64-bit cores (CVA6):** `uint64_t` is used as-is.
- **32-bit cores (Snitch):** The lower 32 bits are extracted via `BINGO_LOW32()`.

Both architectures can initialize, allocate from, and free into the same
heap instance in shared memory (TCDM or SPM).

## API

```c
#include "bingo_alloc.h"

// Initialize a heap in a memory region.
uint64_t handle = bingoHeapInit(base_addr, size);

// Allocate a block.
uint64_t ptr = bingoHeapMalloc(handle, num_bytes);

// Free a block (coalesces with neighbours).
bingoHeapFree(handle, ptr);

// Read diagnostics (capacity, allocated, peak, oom_count).
BingoHeapDiagnostics *d =
    (BingoHeapDiagnostics *)(uintptr_t)bingoHeapGetDiagnostics(handle);
```

## Files

```
bingo_alloc/
├── README.md                  (this file)
└── bingo_alloc/
    ├── bingo_alloc.h          (types + API declarations)
    └── bingo_alloc.c          (implementation)
```
