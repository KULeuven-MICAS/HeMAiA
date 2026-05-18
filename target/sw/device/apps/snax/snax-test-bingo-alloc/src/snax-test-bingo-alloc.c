// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Xiaoling Yi <xiaoling.yi@kuleuven.be>
//
// Test suite for bingo_alloc from the Snitch device side. This mirrors the
// host-side test_bingo_alloc coverage and uses the same bingoHeap* API, but
// initializes the test arena from the SNRT L1 heap metadata.

#include "snrt.h"

static int tests_passed = 0;
static int tests_failed = 0;

static uint64_t init_l1_test_heap(void) {
    snrt_l1_allocator_t *allocator = get_snrt_l1_allocator();
    allocator->l1_heap_manager =
        bingoHeapInit(allocator->l1_heap_start_addr, allocator->l1_heap_size);
    return allocator->l1_heap_manager;
}

static BingoHeapDiagnostics *l1_diagnostics(uint64_t heap) {
    return (BingoHeapDiagnostics *)(uintptr_t)bingoHeapGetDiagnostics(heap);
}

static uint64_t expected_allocated_size(uint64_t amount) {
    const uint64_t fragment_size_min = (uint64_t)BINGO_HEAP_ALIGNMENT * 2U;
    return ((uint64_t)amount + BINGO_HEAP_ALIGNMENT + fragment_size_min - 1U) &
           ~(fragment_size_min - 1U);
}

#define TEST_ASSERT(cond, msg)                                         \
    do {                                                               \
        if (!(cond)) {                                                 \
            printf("  FAIL: %s (line %d)\r\n", msg, __LINE__);        \
            tests_failed++;                                            \
        } else {                                                       \
            printf("  PASS: %s\r\n", msg);                            \
            tests_passed++;                                            \
        }                                                              \
    } while (0)

#define TEST_REQUIRE(cond, msg)                                        \
    do {                                                               \
        TEST_ASSERT(cond, msg);                                        \
        if (!(cond)) {                                                 \
            return;                                                    \
        }                                                              \
    } while (0)

static void test_basic(void) {
    printf("\n[Test 1] Basic init / alloc / free\r\n");
    uint64_t heap = init_l1_test_heap();
    TEST_REQUIRE(heap != 0, "bingoHeapInit returns non-zero handle");

    BingoHeapDiagnostics *d = l1_diagnostics(heap);
    TEST_ASSERT(d->capacity > 0, "capacity > 0");
    TEST_ASSERT(d->allocated == 0, "initially allocated == 0");
    printf("  INFO: capacity = %u bytes (%u KB)\r\n",
           (uint32_t)d->capacity, (uint32_t)d->capacity >> 10);

    uint64_t p1 = bingoHeapMalloc(heap, 1024);
    TEST_ASSERT(p1 != 0, "1 KB alloc succeeds");
    TEST_ASSERT(d->allocated > 0, "allocated > 0 after alloc");
    TEST_ASSERT((p1 % BINGO_HEAP_ALIGNMENT) == 0,
                "returned pointer is 128-byte aligned");

    bingoHeapFree(heap, p1);
    TEST_ASSERT(d->allocated == 0, "allocated == 0 after free");
}

static void test_large_alloc(void) {
    printf("\n[Test 2] Large allocation (> 256 KB) - the key fix\r\n");
    uint64_t heap = init_l1_test_heap();
    TEST_REQUIRE(heap != 0, "bingoHeapInit returns non-zero handle");
    BingoHeapDiagnostics *d = l1_diagnostics(heap);

    const uint64_t request_300kb = 300U * 1024U;
    const uint64_t expected_300kb = expected_allocated_size(request_300kb);
    uint64_t p1 = bingoHeapMalloc(heap, request_300kb);
    TEST_ASSERT(p1 != 0, "300 KB alloc succeeds");
    TEST_ASSERT((p1 % BINGO_HEAP_ALIGNMENT) == 0,
                "returned pointer is 128-byte aligned");
    TEST_ASSERT(d->allocated == expected_300kb,
                "allocated tracks rounded fragment size");
    printf("  INFO: requested = %u bytes, allocated = %u bytes\r\n",
           (uint32_t)request_300kb, (uint32_t)d->allocated);

    bingoHeapFree(heap, p1);
    TEST_ASSERT(d->allocated == 0, "freed 300 KB successfully");

    const uint64_t request_400kb = 400U * 1024U;
    const uint64_t expected_400kb = expected_allocated_size(request_400kb);
    uint64_t p2 = bingoHeapMalloc(heap, request_400kb);
    TEST_ASSERT(p2 != 0, "400 KB alloc succeeds");
    TEST_ASSERT((p2 % BINGO_HEAP_ALIGNMENT) == 0,
                "returned pointer is 128-byte aligned");
    TEST_ASSERT(d->allocated == expected_400kb,
                "allocated tracks rounded fragment size");
    printf("  INFO: requested = %u bytes, allocated = %u bytes\r\n",
           (uint32_t)request_400kb, (uint32_t)d->allocated);

    bingoHeapFree(heap, p2);

    uint64_t max_request = d->capacity - BINGO_HEAP_ALIGNMENT;
    uint64_t p3 = bingoHeapMalloc(heap, max_request);
    TEST_ASSERT(p3 != 0, "near-capacity alloc succeeds");
    TEST_ASSERT((p3 % BINGO_HEAP_ALIGNMENT) == 0,
                "returned pointer is 128-byte aligned");
    TEST_ASSERT(d->allocated == d->capacity,
                "near-capacity alloc uses full heap capacity");
    printf("  INFO: allocated %u / %u bytes\r\n",
           (uint32_t)d->allocated, (uint32_t)d->capacity);

    bingoHeapFree(heap, p3);
    TEST_ASSERT(d->allocated == 0, "freed near-capacity alloc successfully");
}

static void test_coalescing(void) {
    printf("\n[Test 3] Multiple allocs + free coalescing\r\n");
    uint64_t heap = init_l1_test_heap();
    TEST_REQUIRE(heap != 0, "bingoHeapInit returns non-zero handle");
    BingoHeapDiagnostics *d = l1_diagnostics(heap);

    uint64_t pA = bingoHeapMalloc(heap, 64U * 1024U);
    uint64_t pB = bingoHeapMalloc(heap, 64U * 1024U);
    uint64_t pD = bingoHeapMalloc(heap, 128U * 1024U);
    TEST_REQUIRE(pA != 0 && pB != 0 && pD != 0,
                 "3 tensor allocs succeed");
    printf("  INFO: A=0x%x, B=0x%x, D=0x%x, total allocated=%u\r\n",
           (uint32_t)pA, (uint32_t)pB, (uint32_t)pD, (uint32_t)d->allocated);

    bingoHeapFree(heap, pD);
    bingoHeapFree(heap, pB);
    bingoHeapFree(heap, pA);
    TEST_ASSERT(d->allocated == 0, "all freed, allocated == 0");

    uint64_t pLarge = bingoHeapMalloc(heap, 400U * 1024U);
    TEST_ASSERT(pLarge != 0, "400 KB alloc succeeds after coalescing");
    TEST_ASSERT((pLarge % BINGO_HEAP_ALIGNMENT) == 0,
                "returned pointer is 128-byte aligned");
    TEST_ASSERT(d->allocated == expected_allocated_size(400U * 1024U),
                "allocated tracks rounded fragment size after coalescing");
    bingoHeapFree(heap, pLarge);
    TEST_ASSERT(d->allocated == 0,
                "freed large alloc after coalescing successfully");
}

static void test_search_within_bin(void) {
    printf("\n[Test 4] Search-within-bin (non-power-of-2 fragments)\r\n");
    uint64_t heap = init_l1_test_heap();
    TEST_REQUIRE(heap != 0, "bingoHeapInit returns non-zero handle");
    BingoHeapDiagnostics *d = l1_diagnostics(heap);

    uint64_t pA = bingoHeapMalloc(heap, 100U * 1024U);
    uint64_t pB = bingoHeapMalloc(heap, 100U * 1024U);
    uint64_t pC = bingoHeapMalloc(heap, 100U * 1024U);
    uint64_t pD = bingoHeapMalloc(heap, 100U * 1024U);
    TEST_REQUIRE(pA != 0 && pB != 0 && pC != 0 && pD != 0,
                 "4x 100KB allocs succeed");

    bingoHeapFree(heap, pB);
    printf("  INFO: freed B, allocated=%u, available ~%u\r\n",
           (uint32_t)d->allocated,
           (uint32_t)(d->capacity - d->allocated));

    uint64_t pE = bingoHeapMalloc(heap, 80U * 1024U);
    TEST_ASSERT(pE != 0, "80 KB alloc fits in freed slot");

    bingoHeapFree(heap, pA);
    bingoHeapFree(heap, pC);
    bingoHeapFree(heap, pD);
    bingoHeapFree(heap, pE);
    TEST_ASSERT(d->allocated == 0, "all freed after fragmentation test");
}

static void test_oom(void) {
    printf("\n[Test 5] OOM handling\r\n");
    uint64_t heap = init_l1_test_heap();
    TEST_REQUIRE(heap != 0, "bingoHeapInit returns non-zero handle");
    BingoHeapDiagnostics *d = l1_diagnostics(heap);

    uint64_t p1 = bingoHeapMalloc(heap, 400U * 1024U);
    TEST_REQUIRE(p1 != 0, "400 KB alloc succeeds");

    uint64_t p2 = bingoHeapMalloc(heap, 200U * 1024U);
    TEST_ASSERT(p2 == 0, "200 KB alloc correctly fails (OOM)");
    TEST_ASSERT(d->oom_count == 1, "oom_count incremented to 1");

    uint64_t p3 = bingoHeapMalloc(heap, 200U * 1024U);
    TEST_ASSERT(p3 == 0, "200 KB alloc correctly fails (OOM)");
    TEST_ASSERT(d->oom_count == 2, "oom_count incremented to 2");

    bingoHeapFree(heap, p1);

    uint64_t p4 = bingoHeapMalloc(heap, 200U * 1024U);
    TEST_ASSERT(p4 != 0, "200 KB alloc succeeds after free");
    bingoHeapFree(heap, p4);
    TEST_ASSERT(d->allocated == 0, "allocated back to 0 after OOM test");
    TEST_ASSERT(d->oom_count == 2, "oom_count remains at 2");
}

static void test_diagnostics(void) {
    printf("\n[Test 6] Diagnostics accuracy\r\n");
    uint64_t heap = init_l1_test_heap();
    TEST_REQUIRE(heap != 0, "bingoHeapInit returns non-zero handle");
    BingoHeapDiagnostics *d = l1_diagnostics(heap);

    TEST_ASSERT(d->allocated == 0, "initial allocated == 0");
    TEST_ASSERT(d->peak_allocated == 0, "initial peak == 0");
    TEST_ASSERT(d->oom_count == 0, "initial oom_count == 0");

    uint64_t p1 = bingoHeapMalloc(heap, 50U * 1024U);
    uint64_t alloc_after_p1 = d->allocated;
    TEST_ASSERT(alloc_after_p1 > 0, "allocated > 0 after 50KB");

    uint64_t p2 = bingoHeapMalloc(heap, 100U * 1024U);
    uint64_t alloc_after_p2 = d->allocated;
    TEST_ASSERT(alloc_after_p2 > alloc_after_p1,
                "allocated grows with second alloc");

    bingoHeapFree(heap, p1);
    TEST_ASSERT(d->allocated < alloc_after_p2,
                "allocated decreases after free");
    TEST_ASSERT(d->peak_allocated == alloc_after_p2,
                "peak_allocated preserved");

    bingoHeapFree(heap, p2);
    TEST_ASSERT(d->allocated == 0, "allocated back to 0");
    TEST_ASSERT(d->peak_allocated == alloc_after_p2,
                "peak still at high-water mark");

    uint64_t p3 = bingoHeapMalloc(heap, 10U);
    TEST_ASSERT(d->peak_request_size == 100U * 1024U,
                "peak_request_size == 100 KB (largest request)");

    bingoHeapFree(heap, p3);
    TEST_ASSERT(d->allocated == 0,
                "allocated back to 0 after diagnostics test");
}

static void test_edge_cases(void) {
    printf("\n[Test 7] Edge cases\r\n");
    uint64_t heap = init_l1_test_heap();
    TEST_REQUIRE(heap != 0, "bingoHeapInit returns non-zero handle");
    BingoHeapDiagnostics *d = l1_diagnostics(heap);

    uint64_t p0 = bingoHeapMalloc(heap, 0);
    TEST_ASSERT(p0 == 0, "0-byte alloc returns 0");
    TEST_ASSERT(d->oom_count == 0, "0-byte alloc does not increment oom_count");

    bingoHeapFree(heap, 0);
    TEST_ASSERT(d->allocated == 0, "free(0) is a safe no-op");

    uint64_t p1 = bingoHeapMalloc(heap, 1);
    TEST_ASSERT(p1 != 0, "1-byte alloc succeeds");
    TEST_ASSERT((p1 % BINGO_HEAP_ALIGNMENT) == 0,
                "1-byte alloc is aligned");
    bingoHeapFree(heap, p1);

    uint8_t tiny[64] __attribute__((aligned(BINGO_HEAP_ALIGNMENT)));
    uint64_t bad = bingoHeapInit((uint64_t)(uintptr_t)tiny, 64);
    TEST_ASSERT(bad == 0, "init with 64-byte region fails gracefully");
}

int main() {
    if (snrt_cluster_idx() != 0 || !snrt_is_dm_core()) {
        return 0;
    }

    snrt_l1_allocator_t *allocator = get_snrt_l1_allocator();
    printf("=== bingo_alloc Snitch Test Suite ===\r\n");
    printf("Cluster: %u, L1 heap size: %u KB, BINGO_HEAP_ALIGNMENT: %u\r\n",
           snrt_cluster_idx(), (uint32_t)allocator->l1_heap_size >> 10,
           BINGO_HEAP_ALIGNMENT);

    test_basic();
    test_large_alloc();
    test_coalescing();
    test_search_within_bin();
    test_oom();
    test_diagnostics();
    test_edge_cases();

    printf("\n=== Results: %d passed, %d failed ===\r\n",
           tests_passed, tests_failed);
    if (tests_failed > 0) {
        printf("SOME TESTS FAILED!\r\n");
        return 1;
    }
    printf("ALL TESTS PASSED!\r\n");
    return 0;
}
