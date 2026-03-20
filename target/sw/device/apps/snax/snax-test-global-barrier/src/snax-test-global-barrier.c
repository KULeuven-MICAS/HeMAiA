// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>

#include "snrt.h"

int main() {
    int time_start, time_end;
    
    if (snrt_cluster_idx() == 0 && snrt_is_dm_core()) {
        printf_safe("Cluster 0 DM core before global barrier.\r\n");
    }
    time_start = snrt_mcycle();
    snrt_global_barrier();
    time_end = snrt_mcycle();
    printf_safe("[Cluster %d Core %d] Global barrier latency is %d cycles.\r\n", snrt_cluster_idx(), snrt_cluster_core_idx(), time_end - time_start);
    return 0;
}
