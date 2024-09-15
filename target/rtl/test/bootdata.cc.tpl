// Copyright 2021 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

#include <tb_lib.hh>

namespace sim {

<%
    mem_offset = single_chip_id << (addr_width - chip_id_width)
%>

const BootData BOOTDATA = {.boot_addr = ${hex(boot_addr + mem_offset)},
                           .core_count = ${core_count},
                           .hartid_base = ${hart_id_base},
                           .tcdm_start = ${hex(tcdm_start + mem_offset)},
                           .tcdm_size = ${hex(tcdm_size)},
                           .tcdm_offset = ${hex(tcdm_offset)},
                           .global_mem_start = ${hex(global_mem_start + mem_offset)},
                           .global_mem_end = ${hex(global_mem_end + mem_offset)},
                           .cluster_count = ${cluster_count},
                           .s1_quadrant_count = ${s1_quadrant_count},
                           .clint_base = ${hex(clint_base + mem_offset)}
                        };

}  // namespace sim
