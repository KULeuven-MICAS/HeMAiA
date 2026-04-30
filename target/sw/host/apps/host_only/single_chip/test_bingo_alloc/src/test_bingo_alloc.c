// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
//
// Test suite for bingo_alloc: validates the modified allocator that uses
// alignment-based rounding (not power-of-2), enabling allocations > capacity/2.
//
// This test runs on the CVA6 host core using a static buffer to simulate TCDM.
// No actual TCDM or cluster hardware is required.

#include "host.h"
#include "bingo_alloc.h"

// Simulated TCDM: 512 KB, 128-byte aligned
#define TCDM_SIZE (512 * 1024)
static uint8_t __attribute__((aligned(BINGO_HEAP_ALIGNMENT))) fake_tcdm[TCDM_SIZE];

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(cond, msg)                                          \
    do {                                                                \
        if (!(cond)) {                                                  \
            printf("  FAIL: %s (line %d)\r\n", msg, __LINE__);         \
            tests_failed++;                                             \
        } else {                                                        \
            printf("  PASS: %s\r\n", msg);                             \
            tests_passed++;                                             \
        }                                                               \
    } while (0)

// ---- Test 1: Basic init + alloc + free ----
static void test_basic(void) {
    printf("\n[Test 1] Basic init / alloc / free\r\n");
    uint64_t heap = bingoHeapInit((uint64_t)(uintptr_t)fake_tcdm, TCDM_SIZE);
    TEST_ASSERT(heap != 0, "bingoHeapInit returns non-zero handle");

    BingoHeapDiagnostics *d = (BingoHeapDiagnostics *)(uintptr_t)bingoHeapGetDiagnostics(heap);
    TEST_ASSERT(d->capacity > 0, "capacity > 0");
    TEST_ASSERT(d->allocated == 0, "initially allocated == 0");
    printf("  INFO: capacity = %lu bytes (%lu KB)\r\n", d->capacity, d->capacity / 1024);

    // Allocate 1 KB
    uint64_t p1 = bingoHeapMalloc(heap, 1024);
    TEST_ASSERT(p1 != 0, "1 KB alloc succeeds");
    TEST_ASSERT(d->allocated > 0, "allocated > 0 after alloc");
    TEST_ASSERT((p1 % BINGO_HEAP_ALIGNMENT) == 0, "returned pointer is 128-byte aligned");

    // Free
    bingoHeapFree(heap, p1);
    TEST_ASSERT(d->allocated == 0, "allocated == 0 after free");
}

// ---- Test 2: Allocate > 256 KB (the key fix) ----
static void test_large_alloc(void) {
    printf("\n[Test 2] Large allocation (> 256 KB) — the key fix\r\n");
    uint64_t heap = bingoHeapInit((uint64_t)(uintptr_t)fake_tcdm, TCDM_SIZE);
    BingoHeapDiagnostics *d = (BingoHeapDiagnostics *)(uintptr_t)bingoHeapGetDiagnostics(heap);

    // 300 KB — would FAIL with o1heap (rounds to 512 KB)
    uint64_t p1 = bingoHeapMalloc(heap, 300 * 1024);
    TEST_ASSERT(p1 != 0, "300 KB alloc succeeds (would fail with o1heap)");
    printf("  INFO: allocated = %lu bytes after 300 KB request\r\n", d->allocated);

    bingoHeapFree(heap, p1);
    TEST_ASSERT(d->allocated == 0, "freed 300 KB successfully");

    // 400 KB — also should work
    uint64_t p2 = bingoHeapMalloc(heap, 400 * 1024);
    TEST_ASSERT(p2 != 0, "400 KB alloc succeeds");
    printf("  INFO: allocated = %lu bytes after 400 KB request\r\n", d->allocated);

    bingoHeapFree(heap, p2);

    // Near-capacity: capacity - 256 (one FRAGMENT_SIZE_MIN for metadata)
    uint64_t max_request = d->capacity - BINGO_HEAP_ALIGNMENT;
    uint64_t p3 = bingoHeapMalloc(heap, max_request);
    TEST_ASSERT(p3 != 0, "near-capacity alloc succeeds");
    printf("  INFO: allocated %lu / %lu bytes (%.1f%%)\r\n",
           d->allocated, d->capacity, 100.0 * d->allocated / d->capacity);

    bingoHeapFree(heap, p3);
}

// ---- Test 3: Multiple allocations + coalescing ----
static void test_coalescing(void) {
    printf("\n[Test 3] Multiple allocs + free coalescing\r\n");
    uint64_t heap = bingoHeapInit((uint64_t)(uintptr_t)fake_tcdm, TCDM_SIZE);
    BingoHeapDiagnostics *d = (BingoHeapDiagnostics *)(uintptr_t)bingoHeapGetDiagnostics(heap);

    // Allocate 3 chunks: simulating A (64KB), B (64KB), D (128KB) tensors
    uint64_t pA = bingoHeapMalloc(heap, 64 * 1024);
    uint64_t pB = bingoHeapMalloc(heap, 64 * 1024);
    uint64_t pD = bingoHeapMalloc(heap, 128 * 1024);
    TEST_ASSERT(pA != 0 && pB != 0 && pD != 0, "3 tensor allocs succeed");
    printf("  INFO: A=0x%lx, B=0x%lx, D=0x%lx, total allocated=%lu\r\n",
           pA, pB, pD, d->allocated);

    // Free in reverse order (best for coalescing)
    bingoHeapFree(heap, pD);
    bingoHeapFree(heap, pB);
    bingoHeapFree(heap, pA);
    TEST_ASSERT(d->allocated == 0, "all freed, allocated == 0");

    // After coalescing, should be able to allocate a large block again
    uint64_t pLarge = bingoHeapMalloc(heap, 400 * 1024);
    TEST_ASSERT(pLarge != 0, "400 KB alloc succeeds after coalescing");
    bingoHeapFree(heap, pLarge);
}

// ---- Test 4: Fragmentation + search-within-bin ----
static void test_search_within_bin(void) {
    printf("\n[Test 4] Search-within-bin (non-power-of-2 fragments)\r\n");
    uint64_t heap = bingoHeapInit((uint64_t)(uintptr_t)fake_tcdm, TCDM_SIZE);
    BingoHeapDiagnostics *d = (BingoHeapDiagnostics *)(uintptr_t)bingoHeapGetDiagnostics(heap);

    // Create fragmentation: alloc A(100KB), B(100KB), C(100KB)
    uint64_t pA = bingoHeapMalloc(heap, 100 * 1024);
    uint64_t pB = bingoHeapMalloc(heap, 100 * 1024);
    uint64_t pC = bingoHeapMalloc(heap, 100 * 1024);
    TEST_ASSERT(pA != 0 && pB != 0 && pC != 0, "3x 100KB allocs succeed");

    // Free B (middle) — creates a ~100KB hole between A and C
    bingoHeapFree(heap, pB);
    printf("  INFO: freed B, allocated=%lu, available ~%lu\r\n",
           d->allocated, d->capacity - d->allocated);

    // Allocate 80 KB — should fit in B's former slot
    // This exercises search-within-bin: the ~100KB fragment and the
    // remaining tail fragment may be in the same bin
    uint64_t pD = bingoHeapMalloc(heap, 80 * 1024);
    TEST_ASSERT(pD != 0, "80 KB alloc fits in freed slot");

    // Free everything
    bingoHeapFree(heap, pA);
    bingoHeapFree(heap, pC);
    bingoHeapFree(heap, pD);
    TEST_ASSERT(d->allocated == 0, "all freed after fragmentation test");
}

// ---- Test 5: OOM handling ----
static void test_oom(void) {
    printf("\n[Test 5] OOM handling\r\n");
    uint64_t heap = bingoHeapInit((uint64_t)(uintptr_t)fake_tcdm, TCDM_SIZE);
    BingoHeapDiagnostics *d = (BingoHeapDiagnostics *)(uintptr_t)bingoHeapGetDiagnostics(heap);

    // Fill most of the heap
    uint64_t p1 = bingoHeapMalloc(heap, 400 * 1024);
    TEST_ASSERT(p1 != 0, "400 KB alloc succeeds");

    // Try to alloc more than remaining — should fail gracefully
    uint64_t p2 = bingoHeapMalloc(heap, 200 * 1024);
    TEST_ASSERT(p2 == 0, "200 KB alloc correctly fails (OOM)");
    TEST_ASSERT(d->oom_count == 1, "oom_count incremented to 1");

    bingoHeapFree(heap, p1);

    // After free, the alloc should succeed
    uint64_t p3 = bingoHeapMalloc(heap, 200 * 1024);
    TEST_ASSERT(p3 != 0, "200 KB alloc succeeds after free");
    bingoHeapFree(heap, p3);
}

// ---- Test 6: Diagnostics accuracy ----
static void test_diagnostics(void) {
    printf("\n[Test 6] Diagnostics accuracy\r\n");
    uint64_t heap = bingoHeapInit((uint64_t)(uintptr_t)fake_tcdm, TCDM_SIZE);
    BingoHeapDiagnostics *d = (BingoHeapDiagnostics *)(uintptr_t)bingoHeapGetDiagnostics(heap);

    TEST_ASSERT(d->allocated == 0, "initial allocated == 0");
    TEST_ASSERT(d->peak_allocated == 0, "initial peak == 0");
    TEST_ASSERT(d->oom_count == 0, "initial oom_count == 0");

    uint64_t p1 = bingoHeapMalloc(heap, 50 * 1024);
    uint64_t alloc_after_p1 = d->allocated;
    TEST_ASSERT(alloc_after_p1 > 0, "allocated > 0 after 50KB");

    uint64_t p2 = bingoHeapMalloc(heap, 100 * 1024);
    uint64_t alloc_after_p2 = d->allocated;
    TEST_ASSERT(alloc_after_p2 > alloc_after_p1, "allocated grows with second alloc");

    bingoHeapFree(heap, p1);
    TEST_ASSERT(d->allocated < alloc_after_p2, "allocated decreases after free");
    TEST_ASSERT(d->peak_allocated == alloc_after_p2, "peak_allocated preserved");

    bingoHeapFree(heap, p2);
    TEST_ASSERT(d->allocated == 0, "allocated back to 0");
    TEST_ASSERT(d->peak_allocated == alloc_after_p2, "peak still at high-water mark");

    // peak_request_size tracks the largest request
    bingoHeapMalloc(heap, 10);
    TEST_ASSERT(d->peak_request_size == 100 * 1024, "peak_request_size == 100 KB (largest request)");
}

// ---- Test 7: Zero and edge-case requests ----
static void test_edge_cases(void) {
    printf("\n[Test 7] Edge cases\r\n");
    uint64_t heap = bingoHeapInit((uint64_t)(uintptr_t)fake_tcdm, TCDM_SIZE);
    BingoHeapDiagnostics *d = (BingoHeapDiagnostics *)(uintptr_t)bingoHeapGetDiagnostics(heap);

    // Zero-size allocation should return 0 (no-op)
    uint64_t p0 = bingoHeapMalloc(heap, 0);
    TEST_ASSERT(p0 == 0, "0-byte alloc returns 0");
    TEST_ASSERT(d->oom_count == 0, "0-byte alloc does not increment oom_count");

    // Free of NULL pointer should be no-op
    bingoHeapFree(heap, 0);
    TEST_ASSERT(d->allocated == 0, "free(0) is a safe no-op");

    // Minimum allocation (1 byte)
    uint64_t p1 = bingoHeapMalloc(heap, 1);
    TEST_ASSERT(p1 != 0, "1-byte alloc succeeds");
    TEST_ASSERT((p1 % BINGO_HEAP_ALIGNMENT) == 0, "1-byte alloc is aligned");
    bingoHeapFree(heap, p1);

    // Init with too-small region should fail
    uint8_t tiny[64] __attribute__((aligned(BINGO_HEAP_ALIGNMENT)));
    uint64_t bad = bingoHeapInit((uint64_t)(uintptr_t)tiny, 64);
    TEST_ASSERT(bad == 0, "init with 64-byte region fails gracefully");
}

int main() {
    uintptr_t address_prefix = (uintptr_t)get_current_chip_baseaddress();
    init_uart(address_prefix, 32, 1);
    enable_vec();
    asm volatile("fence" : : : "memory");

    printf("=== bingo_alloc Test Suite ===\r\n");
    printf("TCDM size: %d KB, BINGO_HEAP_ALIGNMENT: %d\r\n",
           TCDM_SIZE / 1024, BINGO_HEAP_ALIGNMENT);

    test_basic();
    test_large_alloc();
    test_coalescing();
    test_search_within_bin();
    test_oom();
    test_diagnostics();
    test_edge_cases();

    printf("\n=== Results: %d passed, %d failed ===\r\n", tests_passed, tests_failed);
    if (tests_failed > 0) {
        printf("SOME TESTS FAILED!\r\n");
        return 1;
    }
    printf("ALL TESTS PASSED!\r\n");
    return 0;
}
