// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
#include "snrt.h"
int main(){
    // First we need to write the kernel tab to the soc ctrl
    // We use the dm core of first cluster in first chiplet to do this
    if ((get_current_chip_id() == 0) & (snrt_cluster_idx() == 0)){
        if(snrt_is_dm_core()){
            // printf("[Cluster %d] DM core is writing the kernel tab to the soc ctrl\n", snrt_cluster_idx());
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
    // Each chiplet will start with this function and wait for the host to offload tasks
    // For now we use a centralized way:
    //      The host in chiplet 0 will handle all the offloaded tasks
    //      The host in chiplet 1 will first initialize the mailbox and then put itself to sleep
    return bingo_offload_manager();
}
