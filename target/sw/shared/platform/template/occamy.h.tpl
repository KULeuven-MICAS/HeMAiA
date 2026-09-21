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

// Task-id width of the bingo HW manager (s1_quadrant.task_id_width -> TaskIdWidth). The id
// space is 2**width and the mini-compiler hands out one id per task, so a graph with more
// tasks than this has nowhere to put them. It is also a descriptor field, so SW and RTL must
// agree or every field above task_id shifts.
#define BINGO_TASK_ID_WIDTH            ${task_id_width}

// Packed task-descriptor width of the bingo HW manager (bingo_hw_manager_top
// TaskDescBusWidth). DERIVED per config by occamygen -- the smallest whole number of
// 64-bit words that holds this config's descriptor layout -- unless the cfg pins it wider
// with s1_quadrant.task_desc_width. The descriptor no longer has to fit one 64-bit
// host AXI-Lite beat: the task-queue master fetches BINGO_TASK_DESC_WORDS beats and
// commits them as one atomic push. The task list stays a uint64_t array with WORDS
// entries per descriptor, least-significant word FIRST (beat 0 is read from the lower
// address into the low bits), so the descriptor stride is BINGO_TASK_DESC_WIDTH / 8
// bytes. Both the C packer and the mini-compiler must read these, never a literal 64.
#define BINGO_TASK_DESC_WIDTH          ${task_desc_width}
#define BINGO_TASK_DESC_WORDS          ${task_desc_words}

// Number of cores the bingo HW manager sees per cluster: N_CORES_PER_CLUSTER counts only
// the snitch cores, but the host CVA6 is wired in as one extra core of cluster 0, so the
// RTL elaborates NUM_CORES_PER_CLUSTER = N_CORES_PER_CLUSTER + 1 (occamy_quad_ctrl.sv).
// The descriptor's dep_check_code / dep_set_code bitmaps and the assigned_core_id field
// are sized from THIS number. Software used to re-derive the +1 by hand (or forget it,
// which silently shifted every field above assigned_core_id) -- use this define instead.
#define BINGO_NCORES_HW                ${bingo_ncores_hw}

// D2D routing-id width of the bingo HW manager (hemaia_multichip.chip_id_width ->
// bingo_hw_manager_top ChipIdWidth, occamy_pkg ChipIdWidth / chip_id_t). BOTH descriptor
// chiplet-id fields -- assigned_chiplet_id and dep_set_chiplet_id -- are this wide,
// because they carry the D2D routing id ((x << 4) | y) and NOT an index into N_CHIPLETS:
// sizing them from N_CHIPLETS_WIDTH is the bug this define exists to prevent, and it also
// shifts every descriptor field above them. bingo_utils.h picks this up instead of its
// #ifndef fallback of 8.
#define BINGO_CHIP_ID_WIDTH            ${chip_id_width}
