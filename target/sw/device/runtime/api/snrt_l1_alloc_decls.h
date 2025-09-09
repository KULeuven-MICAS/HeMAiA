// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>

typedef struct {
    O1HeapInstance *l1_heap_manager;
    uint32_t l1_heap_start_addr;
    uint32_t l1_heap_size;
} snrt_l1_allocator_t;

// inline uint32_t snrt_l1_malloc_init();
inline uint32_t *snrt_l1_malloc(uint32_t size);
inline void snrt_l1_free(uint32_t *addr);