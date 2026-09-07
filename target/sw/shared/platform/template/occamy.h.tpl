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

// Compute-chiplet grid extents, derived from the cfg's hemaia_compute_chip coordinates
// (max(coord) + 1 on each axis). chip_id = (x << 4) | y, so a chip's position on the
// virtual interposer is (chip_id >> 4, chip_id & 0xF) and these say where the array ends.
// The D2D link-availability programming needs them to know which of its four PHY ports
// face a real neighbour and which face off-array.
#define N_CHIPLETS_X                   ${nr_chiplets_x}
#define N_CHIPLETS_Y                   ${nr_chiplets_y}

// Memory chiplet placement. The testharness requires the memchip to sit on exactly one
// edge of the compute array, so MEM_CHIP_LOC_X == N_CHIPLETS_X means it hangs off the
// EAST port of compute chip (N_CHIPLETS_X - 1, MEM_CHIP_LOC_Y). N_MEM_CHIPS is 0 when the
// cfg declares none, in which case the LOC values are meaningless.
#define N_MEM_CHIPS                    ${nr_mem_chips}
#define MEM_CHIP_LOC_X                 ${mem_chip_loc_x}
#define MEM_CHIP_LOC_Y                 ${mem_chip_loc_y}

// Whether the testharness memchip clock runs at the same speed as the host clock.
#define HEMAIA_SAME_MEMCHIP_SPEED      ${same_memchip_speed}

// CLINT MSIP bit the bingo HW manager writes to ring the host DVFS doorbell (a
// dedicated interrupt target appended after this chiplet's harts). Keep in sync with
// occamy.py hw_manager_ipi_idx / occamy_soc.sv.tpl.
#define HW_MANAGER_DVFS_MSIP_BIT       ${hw_manager_dvfs_msip_bit}

// Per-edge dependency tag width of the bingo HW manager (s1_quadrant.dep_tag_width ->
// bingo_hw_manager_top DepTagWidth). The task descriptor packs one tag at the MSB of
// each dep_*_info field, so SW must use the same width as the RTL: bingo_utils.h derives
// DEP_TAG_WIDTH from this, and the mini-compiler passes it to BingoDFG(dep_tag_width=).
#define BINGO_DEP_TAG_WIDTH            ${dep_tag_width}
