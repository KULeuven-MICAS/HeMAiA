// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#define N_CHIPLETS ${nr_chiplets}
% for idx, chiplet_id in enumerate(chiplet_ids):
#define CHIPLET_ID_${idx} 0x${f"{chiplet_id:02x}"}
% endfor
#define N_CLUSTERS ${nr_clusters}
#define N_SNITCHES ${nr_cores}

#define N_CHIPLETS_WIDTH               ${clog2_nr_chiplets}
#define N_CLUSTERS_PER_CHIPLET         ${nr_clusters_per_chiplet}
#define N_CLUSTERS_PER_CHIPLET_WIDTH   ${clog2_nr_clusters_per_chiplet}
#define N_CORES_PER_CLUSTER            ${nr_cores_per_cluster}
#define N_CORES_PER_CLUSTER_WIDTH      ${clog2_nr_cores_per_cluster}

// Whether the testharness memchip clock runs at the same speed as the host clock.
#define HEMAIA_SAME_MEMCHIP_SPEED      ${same_memchip_speed}

#define BINGO_SCHEDULER_ENABLED        ${int(bingo_enabled)}
#define BINGO_PROTOCOL_NUM_CLUSTERS    ${bingo_num_clusters}
#define BINGO_PROTOCOL_NUM_CORES       ${bingo_num_cores_per_cluster}
#define BINGO_PROTOCOL_CHIP_ID_WIDTH   ${bingo_chip_id_width}
#define BINGO_PROTOCOL_CLUSTER_ID_WIDTH ${max(1, (bingo_num_clusters - 1).bit_length())}
#define BINGO_PROTOCOL_CORE_ID_WIDTH   ${max(1, (bingo_num_cores_per_cluster - 1).bit_length())}
#define BINGO_HOST_CORE_ID             (BINGO_PROTOCOL_NUM_CORES - 1)
#define BINGO_DVFS_ENABLED             ${int(bingo_enable_dvfs)}
% if bingo_enable_dvfs:
// CLINT MSIP bit the Bingo HW manager uses to notify the host about DVFS.
#define HW_MANAGER_DVFS_MSIP_BIT       ${hw_manager_dvfs_msip_bit}
% endif

// Per-edge tag width of the Bingo wire descriptor.
#define BINGO_DEP_TAG_WIDTH            ${dep_tag_width}
