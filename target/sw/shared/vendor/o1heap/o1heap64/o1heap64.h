
#ifndef O1HEAP_H_INCLUDED
#define O1HEAP_H_INCLUDED
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/// The guaranteed alignment depends on the platform pointer width.
#define O1HEAP_ALIGNMENT 64  // Force 64-byte alignment for all systems.
#define NUM_BINS_MAX 64

/// Runtime diagnostic information. This information can be used to facilitate runtime self-testing,
/// as required by certain safety-critical development guidelines.
typedef struct __attribute__((aligned(8)))
{
    uint64_t capacity;        ///< Total memory available for allocation (in bytes).
    uint64_t allocated;       ///< Currently allocated memory (in bytes).
    uint64_t peak_allocated;  ///< Peak memory allocated since initialization.
    uint64_t peak_request_size; ///< Largest allocation request (in bytes).
    uint64_t oom_count;       ///< Out-of-memory error count.
} O1HeapDiagnostics;

/// The heap instance structure.
typedef struct __attribute__((aligned(8))) FragmentHeader
{
    uint64_t next;
    uint64_t prev;
    uint64_t size;
    uint64_t used;
} FragmentHeader;

typedef struct __attribute__((aligned(8))) Fragment
{
    FragmentHeader header;
    uint64_t next_free;
    uint64_t prev_free;
} Fragment; 

typedef struct __attribute__((aligned(8))) O1HeapInstance
{
    uint64_t __attribute__((aligned(8))) bins[NUM_BINS_MAX];  ///< Array of bins, each storing a pointer as uint64_t.
    uint64_t nonempty_bin_mask;   ///< Bitmask indicating non-empty bins.
    O1HeapDiagnostics diagnostics; ///< Diagnostic information.
} O1HeapInstance;
/// Initializes a new heap instance in the provided memory arena.
/// The arena base pointer shall be aligned at O1HEAP_ALIGNMENT.
/// Returns a pointer to the initialized heap instance, or NULL on failure.
uint64_t o1heapInit(uint64_t const heap_base_addr, uint64_t const heap_size);

/// Allocates a memory block of the specified size from the heap.
/// Returns a pointer to the allocated memory, or NULL if the allocation fails.
uint64_t o1heapAllocate(uint64_t const handle_ptr_val, const uint64_t amount);

/// Frees a previously allocated memory block.
/// The pointer must have been returned by o1heapAllocate.
void o1heapFree(uint64_t const handle_ptr_val, uint64_t const pointer_val);

/// Returns a copy of the diagnostic information for the heap instance.
uint64_t o1heapGetDiagnostics(uint64_t const handle_ptr_val);

#ifdef __cplusplus
}   
#endif
#endif  // O1HEAP_H_INCLUDED
