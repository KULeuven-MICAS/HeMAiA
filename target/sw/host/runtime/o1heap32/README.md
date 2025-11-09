# o1heap32: 32‑bit deterministic heap on a 64‑bit host

o1heap32 is a host-side variant of o1heap that enforces 32-bit semantics for sizes, indices, and pointer storage, even when compiled with a 64-bit compiler. It is intended to interoperate with a 32-bit device heap (e.g., SNAX cluster L1 memory) from a 64-bit host while preserving identical data layout and behavior.

Key goals:
- All sizes and indices use uint32_t (no size_t).
- All stored “pointers” in heap metadata are 32-bit wide.
- API and types are suffixed with _32 to avoid collisions with the original o1heap.
- Alignment and binning logic follow the original o1heap, adapted for 32-bit arithmetic.

Note: In this host port, heap initialization is typically performed on the device. The host receives a pointer to an already-initialized heap instance and uses o1heap32 APIs to allocate/free blocks in that same arena.

## Usage pattern (host)

- Obtain a handle to the device-initialized heap instance.
- Allocate and free using the _32 APIs:
```c
// Example
O1HeapInstance32* h = bingo_get_l1_heap_manager(chip_id, cluster_id);
uint32_t sz = 4096;
void* p = o1heapAllocate32(h, sz);
if (p) { /* use p */ }
o1heapFree32(h, p);
```
