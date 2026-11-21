# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51

proc hemaia_configure_vpk180_dram_pmc {cips} {
    set_property CONFIG.PS_PMC_CONFIG(PMC_USE_PMC_NOC_AXI0) 1 $cips
}

proc hemaia_connect_vpk180_dram {axi_pin clock_pin cips} {
    set diagram [current_bd_design]

    # UG1582 and the Vivado Versal Common Platform use one DDRMC per
    # VPK180 triplet. Apply the board presets for the Micron LPDDR4,
    # 200 MHz differential clocks and all package pin constraints.
    foreach {triplet region0 region1} {
        3 DDR_LOW0 DDR_LOW1
        2 DDR_CH1 NONE
        1 DDR_CH2 NONE
    } {
        set mc [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_noc:1.1 dram_triplet_$triplet]
        foreach {board_interface pin} [list \
            ch0_lpddr4_trip$triplet CH0_LPDDR4_0 \
            ch1_lpddr4_trip$triplet CH1_LPDDR4_0 \
            lpddr4_clk$triplet sys_clk0 \
        ] {
            apply_board_connection -board_interface $board_interface -ip_intf $mc/$pin -diagram $diagram
        }
        # Distribute adjacent 256-byte NoC packets across the two channels.
        set_property -dict [list \
            CONFIG.NUM_MI 0 \
            CONFIG.MC_CHAN_REGION0 $region0 \
            CONFIG.MC_CHAN_REGION1 $region1 \
            CONFIG.MC_CHANNEL_INTERLEAVING true \
            CONFIG.MC_CH_INTERLEAVING_SIZE 256_Bytes \
        ] $mc
    }

    # The 512-bit PL NMU crosses from the AXI host clock to the hard NoC clock.
    # It reaches the primary DDRMC directly and the other two over the NoC.
    set noc [get_bd_cells dram_triplet_3]
    set_property -dict [list \
        CONFIG.NUM_SI 2 CONFIG.NUM_CLKS 2 CONFIG.NUM_NMI 2 CONFIG.NUM_MCP 3 \
    ] $noc
    set_property CONFIG.ASSOCIATED_BUSIF S00_AXI [get_bd_pins $noc/aclk0]
    set_property CONFIG.ASSOCIATED_BUSIF S01_AXI [get_bd_pins $noc/aclk1]
    connect_bd_net $clock_pin [get_bd_pins $noc/aclk0]
    connect_bd_net [get_bd_pins $cips/pmc_axi_noc_axi0_clk] [get_bd_pins $noc/aclk1]

    # Four 512-bit beats match the NoC's 256-byte packet chopping boundary.
    set pl_qos {read_bw {4000} write_bw {4000} read_avg_burst {4} write_avg_burst {4}}
    set pmc_qos {read_bw {128} write_bw {128} read_avg_burst {4} write_avg_burst {4} initial_boot {true}}
    set pl_paths [dict create MC_0 $pl_qos]
    set pmc_paths [dict create MC_2 $pmc_qos]
    foreach {path triplet} {M00_INI 2 M01_INI 1} {
        set mc [get_bd_cells dram_triplet_$triplet]
        set_property -dict [list CONFIG.NUM_SI 0 CONFIG.NUM_NSI 1 CONFIG.NUM_MCP 1] $mc
        set egress [get_bd_intf_pins $noc/$path]
        set target [get_bd_intf_pins $mc/S00_INI]
        set_property CONFIG.INI_STRATEGY load $egress
        set_property -dict [list \
            CONFIG.INI_STRATEGY load CONFIG.CONNECTIONS [dict create MC_0 $pl_qos] \
        ] $target
        connect_bd_intf_net $egress $target
        dict set pl_paths $path $pl_qos
        dict set pmc_paths $path $pmc_qos
    }

    set_property CONFIG.CONNECTIONS $pl_paths [get_bd_intf_pins $noc/S00_AXI]
    set_property -dict [list CONFIG.CATEGORY ps_pmc CONFIG.CONNECTIONS $pmc_paths] \
        [get_bd_intf_pins $noc/S01_AXI]
    connect_bd_intf_net $axi_pin [get_bd_intf_pins $noc/S00_AXI]
    connect_bd_intf_net [get_bd_intf_pins $cips/PMC_NOC_AXI_0] [get_bd_intf_pins $noc/S01_AXI]

    # Keep these NoC apertures in step with hemaia_dram_address_mapper.sv.
    set pl_space [get_bd_addr_spaces -of_objects $axi_pin]
    set pmc_space [get_bd_addr_spaces $cips/PMC_NOC_AXI_0]
    foreach {pl_target pmc_target offset size} {
        dram_triplet_3/S00_AXI/C0_DDR_LOW0 dram_triplet_3/S01_AXI/C2_DDR_LOW0 0x000000000000 2G
        dram_triplet_3/S00_AXI/C0_DDR_LOW1 dram_triplet_3/S01_AXI/C2_DDR_LOW1 0x000800000000 2G
        dram_triplet_2/S00_INI/C0_DDR_CH1  dram_triplet_2/S00_INI/C0_DDR_CH1  0x050000000000 4G
        dram_triplet_1/S00_INI/C0_DDR_CH2  dram_triplet_1/S00_INI/C0_DDR_CH2  0x060000000000 4G
    } {
        assign_bd_address -target_address_space $pl_space -offset $offset -range $size \
            [get_bd_addr_segs $pl_target]
        assign_bd_address -target_address_space $pmc_space -offset $offset -range $size \
            [get_bd_addr_segs $pmc_target]
    }
}
