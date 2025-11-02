// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>

// inline uint32_t snrt_l1_malloc_init(uint32_t l1_heap_start_addr, uint32_t l1_heap_size) {
//     if(snrt_l1_allocator.l1_heap_manager == NULL) {
//         snrt_l1_allocator.l1_heap_manager = o1heapInit((void *)snrt_l1_allocator.l1_heap_start_addr, l1_heap_size);
//         if (snrt_l1_allocator.l1_heap_manager == NULL) {
//             return -1;
//         } else {
//             snrt_l1_allocator.l1_heap_start_addr = l1_heap_start_addr;
//             snrt_l1_allocator.l1_heap_size = l1_heap_size;
//             return 0;
//         }
//     } else {
//         printf("%s is already initialized\n", __func__);
//     }
// }

#define ALIGN_UP(addr, size) (((addr) + (size)-1) & ~((size)-1))
#define ALIGN_DOWN(addr, size) ((addr) & ~((size)-1))

#define TCDM_ROW_SIZE 512 // 512B per row (8B/Bank*64Bank)
#define RESERVED_SIZE 4096 // reserve 4kB for future use
// Getter functions for L1 allocator
inline snrt_l1_allocator_t *get_snrt_l1_allocator() {
    return (snrt_l1_allocator_t *)&(cls()->l1_allocator);
}


inline uint32_t *snrt_l1_malloc(uint32_t size){
    void *result = o1heapAllocate(get_snrt_l1_allocator()->l1_heap_manager, size);
    if (result==NULL) {
        printf("[Cluster %d] Core(%d) L1 malloc failed for size %d\n", snrt_cluster_idx(), snrt_cluster_core_idx(), size);
        return NULL;
    }
    return (uint32_t *)result;
}
inline void snrt_l1_free(uint32_t *addr){
    o1heapFree(get_snrt_l1_allocator()->l1_heap_manager, (void*)addr);
}

inline void snrt_l1_malloc_init(){
    // Only one core per cluster has to initialize the L1 allocator
    if (snrt_is_dm_core()) {
        uint32_t l1_end_addr = snrt_cluster_base_addrl() + SNRT_TCDM_SIZE;
        extern volatile uint32_t __cdata_start, __cdata_end;
        extern volatile uint32_t __cbss_start, __cbss_end;
        uint32_t cdata_size = ((uint32_t)&__cdata_end) - ((uint32_t)&__cdata_start);
        uint32_t cbss_size = ((uint32_t)&__cbss_end) - ((uint32_t)&__cbss_start);
        uint32_t cls_size = cdata_size + cbss_size;
        uint32_t ret_value_size = 8;
        uint32_t stack_size = snrt_cluster_core_num()*((1 << SNRT_LOG2_STACK_SIZE) + 8);
        uint32_t l1_heap_top_addr = l1_end_addr - cls_size - ret_value_size - stack_size - RESERVED_SIZE;
        uint32_t l1_heap_top_aligned = ALIGN_DOWN(l1_heap_top_addr, TCDM_ROW_SIZE);
        // We put the l1 heap at the start of each cluster tcdm
        get_snrt_l1_allocator()->l1_heap_start_addr = snrt_cluster_base_addrl();
        get_snrt_l1_allocator()->l1_heap_size = l1_heap_top_aligned - snrt_cluster_base_addrl();
        get_snrt_l1_allocator()->l1_heap_manager = o1heapInit((void *)(get_snrt_l1_allocator()->l1_heap_start_addr), get_snrt_l1_allocator()->l1_heap_size);
        // if(snrt_cluster_idx() == 0){
        //     printf("[Cluster0] DM core initialized L1 Allocator\n");
        //     printf("[Cluster0] L1 Alloc Addr 0x%x\n",&(get_snrt_l1_allocator()->l1_heap_manager));
        //     printf("[Cluster0] Heap Start Addr 0x%x\n",get_snrt_l1_allocator()->l1_heap_start_addr);
        //     printf("[Cluster0] Heap Size 0x%x\n",get_snrt_l1_allocator()->l1_heap_size);
        // }
    }
    snrt_cluster_hw_barrier();
}

inline void *snrt_memset(void *ptr, int value, size_t num) {
    for (uint32_t i = 0; i < num; ++i)
        *((uint8_t *)ptr + i) = (unsigned char)value;
    return ptr;
}