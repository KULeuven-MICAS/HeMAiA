#include "stdint.h"

#include "data.h"
#include "memref.h"
#include "snax_rt.h"

/*
 * These libraries are included from github.com/KULeuven-MICAS/snitch_cluster
 * Interested users, might want to look at:
 *
 * /sw/snRuntime/api
 * /target/snitch_cluster/sw/runtime/rtl/src
 * /target/snitch_cluster/sw/runtime/common
 * */
#include <snrt.h>

void _mlir_ciface_run_network(TwoDMemrefI8_t *output, TwoDMemrefI8_t *input);

void _mlir_ciface_snax_debug_gemm(int32_t _ptr_a, int32_t _ptr_b,
                                  int32_t _ptr_c, int32_t when) {
    int8_t *ptr_a, *ptr_b;
    int32_t *ptr_c;
    ptr_a = (int8_t *)_ptr_a;
    ptr_b = (int8_t *)_ptr_b;
    ptr_c = (int32_t *)_ptr_c;

    int thisc = snrt_cluster_core_idx();

    if (thisc == 0) {
        printf("Debugging GeMM at t = %d with A at %p, B at %p, C at %p\r\n",
               when, ptr_a, ptr_b, ptr_c);

        for (int i = 0; i < 5; i++) {
            printf("i%d -> A=%d, B=%d, C=%d\r\n", i, ptr_a[i], ptr_b[i],
                   ptr_c[i]);
        }
    }

    for (uint8_t i = 0; i < 20; i++) {
        if (thisc == i) {
            printf("Core %d present.\r\n", thisc);
            if (snrt_is_dm_core()) {
                printf("I am a dm core\r\n");
            }
        }
        snrt_cluster_hw_barrier();
    }
}

void _mlir_ciface_snax_debug_bias(int32_t _ptr_a, int32_t _ptr_b,
                                  int32_t _ptr_c, int32_t when) {
    int32_t *ptr_a, *ptr_b, *ptr_c;
    ptr_a = (int32_t *)_ptr_a;
    ptr_b = (int32_t *)_ptr_b;
    ptr_c = (int32_t *)_ptr_c;

    int thisc = snrt_cluster_core_idx();
    if (thisc == 0) {
        printf("Debugging bias at t = %d with A at %p, B at %p, C at %p\r\n",
               when, ptr_a, ptr_b, ptr_c);

        for (int i = 0; i < 5; i++) {
            printf("i%d -> A=%d, B=%d, C=%d\r\n", i, ptr_a[i], ptr_b[i],
                   ptr_c[i]);
        }
    }

    for (uint8_t i = 0; i < 20; i++) {
        if (thisc == i) {
            printf("Core %d present.\r\n", thisc);
            if (snrt_is_dm_core()) {
                printf("I am a dm core\r\n");
            }
        }
        snrt_cluster_hw_barrier();
    }
}

void _mlir_ciface_snax_debug_simd(int32_t _ptr_a, int32_t _ptr_b,
                                  int32_t _ptr_c, int32_t when) {
    int32_t *ptr_a;
    int8_t *ptr_c;
    ptr_a = (int32_t *)_ptr_a;
    ptr_c = (int8_t *)_ptr_c;

    int thisc = snrt_cluster_core_idx();
    if (thisc == 0) {
        printf("Debugging SIMD at t = %d with A at %p, C at %p\r\n", when,
               ptr_a, ptr_c);

        for (int i = 0; i < 128; i++) {
            printf("i%d -> A=%d, C=%d\r\n", i, ptr_a[i], ptr_c[i]);
        }
    }
    for (uint8_t i = 0; i < 20; i++) {
        if (thisc == i) {
            printf("Core %d present.\r\n", thisc);
            if (snrt_is_dm_core()) {
                printf("I am a dm core\r\n");
            }
        }
        snrt_cluster_hw_barrier();
    }
}

int main() {
    if (snrt_cluster_idx() == 0) {
        if (snrt_cluster_core_idx() == 0) {
            printf("hello from snitch cluster! running mlperf tiny \r\n");
        }

        // Create memref objects for data stored in L3
        TwoDMemrefI8_t memrefA;
        memrefA.data = &A;
        memrefA.aligned_data = memrefA.data;
        // Shape and Stride need to be defined for dynamic case
        memrefA.shape[0] = 8;
        memrefA.shape[1] = 640;
        memrefA.stride[0] = 640;
        memrefA.stride[1] = 1;
        memrefA.offset = 0;

        TwoDMemrefI8_t memrefB;

        (void)snrt_mcycle();

        _mlir_ciface_run_network(&memrefB, &memrefA);

        snrt_cluster_hw_barrier();

        (void)snrt_mcycle();

        if (snrt_cluster_core_idx() == 0) {
            printf("Got result at %p: \r\n", memrefB.aligned_data);
            int8_t *data = memrefB.aligned_data;
            for (int i = 0; i < 640; i++) {
                printf("%d ", data[i]);
            }
            printf("\r\n");
        }

        snrt_cluster_hw_barrier();

        return 0;
    }
}
