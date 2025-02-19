#include "snrt.h"

int main() {
#pragma GCC unroll 32
    for (int i = 0; i < 32; i++) {
        asm volatile("nop");
    }
    snrt_global_barrier();
    if (snrt_cluster_idx() == 0) {
        if (snrt_is_dm_core()) {
            printf("[C0]D\r\n");
        }
    }
    snrt_global_barrier();
    if (snrt_cluster_idx() == 1) {
        if (snrt_is_dm_core()) {
            printf("[C1]D\r\n");
        }
    }
    snrt_global_barrier();
    if (snrt_cluster_idx() == 2) {
        if (snrt_is_dm_core()) {
            printf("[C2]D\r\n");
        }
    }
    snrt_global_barrier();
    if (snrt_cluster_idx() == 3) {
        if (snrt_is_dm_core()) {
            printf("[C3]D\r\n");
        }
    }
    snrt_global_barrier();
    return 0;
}