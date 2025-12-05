// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>

// Since we are using the host to clear the cluster tcdm memory, we do not need to 
// rely on the cluster core to init the cluster memory
// So we rewrite some functions here to acting like a crt for the clusters

// KEEP IN MIND: All the functions here will be run by all the clusters

#define SNRT_INIT_TLS
#define SNRT_INIT_CLS
#define SNRT_CRT0_CALLBACK2
#define SNRT_INIT_L1_ALLOC


#ifdef SNRT_CRT0_CALLBACK0
    static inline void snrt_crt0_callback0(){
        // Not implemented
    };
#endif


#ifdef SNRT_INIT_TLS
static inline void snrt_init_tls() {
    extern volatile uint32_t __tdata_start, __tdata_end;
    extern volatile uint32_t __tbss_start, __tbss_end;

    size_t size;
    volatile uint32_t tls_ptr;

    // To avoid contentions in main memory, and take advantage of the
    // bandwidth of the DMA, the DM core initializes the TLS section
    // for every core in a cluster.
    if (snrt_is_dm_core()) {
        size = (size_t)(&__tdata_end) - (size_t)(&__tdata_start);

        // First initialize the DM core's .tdata section from main memory
        asm volatile("mv %0, tp" : "=r"(tls_ptr) : :);
        snrt_dma_start_1d((void*)tls_ptr, (void*)(&__tdata_start), size);

        // Then initialize all other cores' .tdata sections from the DM
        // core's. The offset between the TLS section of successive cores
        // is defined in start.S
        size_t tls_offset = (1 << SNRT_LOG2_STACK_SIZE) + 8;
        for (int i = 1; i < snrt_cluster_core_num(); i++) {
            snrt_dma_start_1d((void*)(tls_ptr + i * tls_offset), (void*)tls_ptr,
                              size);
        }
    }

    snrt_cluster_hw_barrier();
}
#endif

#ifdef SNRT_CRT0_CALLBACK1
    static inline void snrt_crt0_callback1(){
        // Not implemented
    };
#endif

#ifdef SNRT_INIT_CLS
static inline uint32_t snrt_cls_base_addr() {
    extern volatile uint32_t __cdata_start, __cdata_end;
    extern volatile uint32_t __cbss_start, __cbss_end;
    uint32_t cdata_size = ((uint32_t)&__cdata_end) - ((uint32_t)&__cdata_start);
    uint32_t cbss_size = ((uint32_t)&__cbss_end) - ((uint32_t)&__cbss_start);
    uint32_t l1_end_addr = snrt_cluster_base_addrl() + SNRT_TCDM_SIZE;
    return l1_end_addr - cdata_size - cbss_size;
}
#endif
#ifdef SNRT_INIT_CLS
static inline void snrt_init_cls() {
    extern volatile uint32_t __cdata_start, __cdata_end;
    extern volatile uint32_t __cbss_start, __cbss_end;

    _cls_ptr = (cls_t*)snrt_cls_base_addr();

    // Only one core per cluster has to do this
    if (snrt_is_dm_core()) {
        void* ptr = (void*)snrt_cls_base_addr();
        size_t size;

        // Copy cdata section to base of the TCDM
        size = (size_t)(&__cdata_end) - (size_t)(&__cdata_start);
        snrt_dma_start_1d(ptr, (void*)(&__cdata_start), size);

        // Clear cbss section
        ptr = (void*)((uint32_t)ptr + size);
        size = (size_t)(&__cbss_end) - (size_t)(&__cbss_start);
        snrt_dma_start_1d(ptr, (void*)(WIDE_ZERO_MEM_BASE_ADDR), size);
    }
}
#endif

#ifdef SNRT_CRT0_CALLBACK2
    static inline void snrt_crt0_callback2(){
        _snrt_cluster_hw_barrier = cluster_hw_barrier_addr(snrt_cluster_idx());
    };
#endif

#ifdef SNRT_INIT_L1_ALLOC
static inline void snrt_init_l1_alloc() { snrt_l1_malloc_init(); }
#endif

#ifdef SNRT_CRT0_CALLBACK3
    static inline void snrt_crt0_callback3(){
        // Not implemented
    };
#endif

#ifdef SNRT_CRT0_CALLBACK4
    static inline void snrt_crt0_callback4(){
        // Not implemented
    };
#endif

static inline uint32_t* snrt_exit_code_destination() {
    return soc_ctrl_scratch_ptr(3);
}
static inline void snrt_exit_default(int exit_code) {
    exit_code = snrt_global_all_to_all_reduction(exit_code);
    if (snrt_global_core_idx() == 0)
        *(snrt_exit_code_destination()) = (exit_code << 1) | 1;
}

static inline void snrt_exit(int exit_code) {
    snrt_exit_default(exit_code);
    // Interrupt the local host to signal the exit code (snitch by default only has the access to local domain)
    if (snrt_global_core_idx() == 0) {
        comm_buffer_t* comm_buffer = get_communication_buffer();
        set_host_sw_interrupt(comm_buffer->chip_id);
    }
}

void snrt_main(){
    int exit_code = 0;

#ifdef SNRT_CRT0_CALLBACK0
    snrt_crt0_callback0();
#endif

#ifdef SNRT_INIT_TLS
    snrt_init_tls();
#endif

#ifdef SNRT_CRT0_CALLBACK1
    snrt_crt0_callback1();
#endif

#ifdef SNRT_INIT_CLS
    snrt_init_cls();
#endif

#ifdef SNRT_CRT0_CALLBACK2
    snrt_crt0_callback2();
#endif

#ifdef SNRT_INIT_L1_ALLOC
    snrt_init_l1_alloc();
#endif

#ifdef SNRT_CRT0_CALLBACK3
    snrt_crt0_callback3();
#endif

    // Invoke the main function
    snrt_cluster_hw_barrier(); // Prebarrier
    extern int main();
    exit_code = main();
    snrt_cluster_hw_barrier(); // Postbarrier
    
#ifdef SNRT_CRT0_CALLBACK4
    snrt_crt0_callback4();
#endif
    snrt_exit(exit_code);
}