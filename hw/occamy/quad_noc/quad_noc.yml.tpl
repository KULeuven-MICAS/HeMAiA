# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
name: ${noc_name}
description: "NW mesh configuration with ${routing_algo} routing for FlooGen"
network_type: "narrow-wide"

routing:
  route_algo: ${routing_algo}
  use_id_table: true
  en_default_idx: true
  default_idx: [0, 0]

protocols:
  - name: "noc_narrow_in"
    type: "narrow"
    protocol: "AXI4"
    data_width: ${narrow_data_width}
    addr_width: ${narrow_addr_width}
    id_width: ${narrow_out_id_width}
    user_width: ${narrow_user_width}
  - name: "noc_narrow_out"
    type: "narrow"
    protocol: "AXI4"
    data_width: ${narrow_data_width}
    addr_width: ${narrow_addr_width}
    id_width: ${narrow_in_id_width}
    user_width: ${narrow_user_width}
  - name: "noc_wide_in"
    type: "wide"
    protocol: "AXI4"
    data_width: ${wide_data_width}
    addr_width: ${wide_addr_width}
    id_width: ${wide_out_id_width}
    user_width: ${wide_user_width}
  - name: "noc_wide_out"
    type: "wide"
    protocol: "AXI4"
    data_width: ${wide_data_width}
    addr_width: ${wide_addr_width}
    id_width: ${wide_in_id_width}
    user_width: ${wide_user_width}

endpoints:
  - name: "cluster"
    array: ${noc_array}
    addr_range:
      base: ${cluster_base_addr}
      size: ${cluster_size}
    mgr_port_protocol:
      - "noc_narrow_in"
      - "noc_wide_in"
    sbr_port_protocol:
      - "noc_narrow_out"
      - "noc_wide_out"

routers:
  - name: "router"
    array: ${noc_array}
    degree: 5

connections:
  - src: "cluster"
    dst: "router"
    src_range:
    - [0, ${noc_array[0]-1}]
    - [0, ${noc_array[1]-1}]
    dst_range:
    - [0, ${noc_array[0]-1}]
    - [0, ${noc_array[1]-1}]
    dst_dir: "Eject"

