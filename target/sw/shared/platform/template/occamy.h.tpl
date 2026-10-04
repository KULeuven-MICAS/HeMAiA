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
// virtual interposer is (chip_id >> 4, chip_id & 0xF). The compute chips need not fill this
// rectangle: a memory chip may sit inside it (a row C M C). Which ports face a chip is
// HEMAIA_D2D_PORT_TABLE below, not these.
#define N_CHIPLETS_X                   ${nr_chiplets_x}
#define N_CHIPLETS_Y                   ${nr_chiplets_y}

// Memory chiplets, in cfg order: MEM_CHIP_ID_<k> = (x << 4) | y. One may sit anywhere a
// D2D link reaches it -- beside the compute array, between compute chips (feeding all of
// them), beside another memory chip. MEM_CHIP_LOC_X/Y is the first one, for code that knows
// only one. N_MEM_CHIPS is 0 when the cfg declares none (the LOC values are then
// meaningless). MEM_CHIP_NUM_SYS_IDMA_<k>: its push engines (sys_dma.h, engine argument).
#define N_MEM_CHIPS                    ${nr_mem_chips}
#define MEM_CHIP_LOC_X                 ${mem_chip_loc_x}
#define MEM_CHIP_LOC_Y                 ${mem_chip_loc_y}
% for k, m in enumerate(mem_chips):
#define MEM_CHIP_ID_${k}                   0x${"%02x" % m["id"]}
#define MEM_CHIP_NUM_SYS_IDMA_${k}         ${m["num_sys_idma"]}
% endfor
% if mem_chips:
#define MEM_CHIP_IDS                   {${", ".join("0x%02x" % m["id"] for m in mem_chips)}}
% endif

// Which chip ids are COMPUTE chips (CHIPLET_ID_<k>): bit (id & 63) of map (id >> 6). A
// rectangle of chip ids -- a chip barrier's -- may hold memory chips, which run no code and
// never arrive; HEMAIA_IS_COMPUTE_CHIP skips them.
<%
    _cmap = [0, 0, 0, 0]
    for _id in chiplet_ids:
        _cmap[_id >> 6] |= 1 << (_id & 63)
%>\
% for _i, _m in enumerate(_cmap):
#define HEMAIA_COMPUTE_CHIP_MAP_${_i}       0x${"%016x" % _m}ULL
% endfor
#define HEMAIA_IS_COMPUTE_CHIP(id) \
    (((((id) >> 6) == 0 ? HEMAIA_COMPUTE_CHIP_MAP_0 : ((id) >> 6) == 1 ? HEMAIA_COMPUTE_CHIP_MAP_1 \
       : ((id) >> 6) == 2 ? HEMAIA_COMPUTE_CHIP_MAP_2 : HEMAIA_COMPUTE_CHIP_MAP_3) \
      >> ((id) & 63)) & 1)

// The D2D ports of every chip on the grid, compute and memory: {chip id, ports that face a
// chip, ports that face a memory chip}, a port being bit d for D2DDirection d (east 0,
// west 1, north 2, south 3). hemaia_d2d_link_initialize_grid() programs the links from it.
// Empty on a single-chip cfg.
#define HEMAIA_D2D_PORT_TABLE_LEN      ${len(d2d_ports)}
#define HEMAIA_D2D_PORT_TABLE          {${", ".join("{0x%02x, 0x%x, 0x%x}" % (p["id"], p["links"], p["mem_links"]) for p in d2d_ports)}}

// Whether the testharness memchip clock runs at the same speed as the host clock.
#define HEMAIA_SAME_MEMCHIP_SPEED      ${same_memchip_speed}
// The simulated clocks (cfg hemaia_multichip.sim_clock): the testbench master clock and the
// divider the runtime gives the host and the clusters. 0 = the runtime's own defaults.
#define HEMAIA_SIM_CLK_MHZ             ${sim_clk_mhz}
#define HEMAIA_CORE_CLK_DIV            ${core_clk_div}

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

// Depth of the bingo HW manager's per-(core, cluster) waiting queue
// (s1_quadrant.bingo_cfg.waiting_queue_depth -> WaitingDepCheckQueueDepth). The in-order
// descriptor stream stops when one of them is full; the mini-compiler's hang check models it.
#define BINGO_WAITING_QUEUE_DEPTH      ${waiting_queue_depth}

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
