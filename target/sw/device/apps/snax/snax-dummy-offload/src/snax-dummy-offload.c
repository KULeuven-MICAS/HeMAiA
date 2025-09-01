// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
#include "snrt.h"
int main(){
    // First we need to write the kernel tab to the soc ctrl
    // We use the first cluster to do this
    if (snrt_cluster_idx() == 0){
        if(snrt_is_dm_core()){
            printf("[Cluster %d] DM core is writing the kernel tab to the soc ctrl\n", snrt_cluster_idx());
            uintptr_t kernel_tab_ready_addr = soc_ctrl_kernel_tab_scratch_addr(0);
            uintptr_t kernel_tab_start_addr = soc_ctrl_kernel_tab_scratch_addr(1);
            uintptr_t kernel_tab_end_addr = soc_ctrl_kernel_tab_scratch_addr(2);
            *((volatile uint32_t *)kernel_tab_start_addr) = (uint32_t)(uintptr_t)&__snax_symtab_start;
            *((volatile uint32_t *)kernel_tab_end_addr)   = (uint32_t)(uintptr_t)&__snax_symtab_end;
            *((volatile uint32_t *)kernel_tab_ready_addr) = 1;
        }
    }
    snrt_global_barrier();
    // Now we can start the offload manager
    // This will be executed by all cores in the cluster

    return bingo_offload_manager();
}
