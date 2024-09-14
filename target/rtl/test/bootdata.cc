// Copyright 2021 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

#include <tb_lib.hh>

namespace sim {

const BootData BOOTDATA = {.boot_addr = 0x010001000000,
                           .core_count = 5,
                           .hartid_base = 1,
                           .tcdm_start = 0x010010000000,
                           .tcdm_size = 0x010000080000,
                           .tcdm_offset = 0x010000100000,
                           .global_mem_start = 0x010080000000,
                           .global_mem_end = 0x010080100000,
                           .cluster_count = 2,
                           .s1_quadrant_count = 1,
                           .clint_base = 0x010004000000};

}  // namespace sim
