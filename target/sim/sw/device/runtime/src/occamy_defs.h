// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "occamy_base_addr.h"
#include "occamy_cfg.h"
#include "snitch_cluster_peripheral.h"

// Hardware parameters
#define SNRT_BASE_HARTID 1
// We do not need this macro since
// the number of cores in each cluster could be different

//#define SNRT_CLUSTER_CORE_NUM N_CORES_PER_CLUSTER
#define SNRT_CLUSTER_NUM (N_QUADS * N_CLUSTERS_PER_QUAD)
#define SNRT_CLUSTER_DM_CORE_NUM 1
#define SNRT_TCDM_START_ADDR QUADRANT_0_CLUSTER_0_TCDM_BASE_ADDR
#define SNRT_TCDM_SIZE                           (QUADRANT_0_CLUSTER_0_PERIPH_BASE_ADDR -      QUADRANT_0_CLUSTER_0_TCDM_BASE_ADDR)
#define SNRT_CLUSTER_OFFSET 0x400000
#define CLUSTER_BASE_ADDR 0x10000000
#define SNRT_CLUSTER_HW_BARRIER_ADDR             (QUADRANT_0_CLUSTER_0_PERIPH_BASE_ADDR +      SNITCH_CLUSTER_PERIPHERAL_HW_BARRIER_REG_OFFSET)

// Software configuration
#define SNRT_LOG2_STACK_SIZE 10

#define CLUSTER_ADDRWIDTH 22
