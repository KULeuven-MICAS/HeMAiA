# This script was generated automatically by bender.
set ROOT "/home/ivean/SNAX"
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/.bender/git/checkouts/tech_cells_generic-8ec0cf131c1245eb/src/fpga/pad_functional_xilinx.sv \
    $ROOT/.bender/git/checkouts/tech_cells_generic-8ec0cf131c1245eb/src/fpga/tc_clk_xilinx.sv \
    $ROOT/.bender/git/checkouts/tech_cells_generic-8ec0cf131c1245eb/src/fpga/tc_sram_xilinx.sv \
    $ROOT/.bender/git/checkouts/tech_cells_generic-8ec0cf131c1245eb/src/rtl/tc_sram_impl.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/.bender/git/checkouts/tech_cells_generic-8ec0cf131c1245eb/src/deprecated/pulp_clock_gating_async.sv \
    $ROOT/.bender/git/checkouts/tech_cells_generic-8ec0cf131c1245eb/src/deprecated/cluster_clk_cells.sv \
    $ROOT/.bender/git/checkouts/tech_cells_generic-8ec0cf131c1245eb/src/deprecated/pulp_clk_cells.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/binary_to_gray.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/cb_filter_pkg.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/cc_onehot.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/cdc_reset_ctrlr_pkg.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/cf_math_pkg.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/clk_int_div.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/delta_counter.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/ecc_pkg.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/edge_propagator_tx.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/exp_backoff.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/fifo_v3.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/gray_to_binary.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/isochronous_4phase_handshake.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/isochronous_spill_register.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/lfsr.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/lfsr_16bit.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/lfsr_8bit.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/lossy_valid_to_stream.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/mv_filter.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/onehot_to_bin.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/plru_tree.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/passthrough_stream_fifo.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/popcount.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/rr_arb_tree.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/rstgen_bypass.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/serial_deglitch.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/shift_reg.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/shift_reg_gated.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/spill_register_flushable.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/stream_demux.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/stream_filter.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/stream_fork.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/stream_intf.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/stream_join_dynamic.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/stream_mux.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/stream_throttle.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/sub_per_hash.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/sync.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/sync_wedge.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/unread.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/read.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/addr_decode_dync.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/cdc_2phase.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/cdc_4phase.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/clk_int_div_static.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/addr_decode.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/addr_decode_napot.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/multiaddr_decode.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/cb_filter.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/cdc_fifo_2phase.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/clk_mux_glitch_free.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/counter.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/ecc_decode.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/ecc_encode.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/edge_detect.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/lzc.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/max_counter.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/rstgen.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/spill_register.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/stream_delay.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/stream_fifo.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/stream_fork_dynamic.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/stream_join.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/cdc_reset_ctrlr.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/cdc_fifo_gray.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/fall_through_register.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/id_queue.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/stream_to_mem.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/stream_arbiter_flushable.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/stream_fifo_optimal_wrap.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/stream_register.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/stream_xbar.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/cdc_fifo_gray_clearable.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/cdc_2phase_clearable.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/mem_to_banks_detailed.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/stream_arbiter.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/stream_omega_net.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/mem_to_banks.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/deprecated/clock_divider_counter.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/deprecated/clk_div.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/deprecated/find_first_one.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/deprecated/generic_LFSR_8bit.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/deprecated/generic_fifo.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/deprecated/prioarbiter.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/deprecated/pulp_sync.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/deprecated/pulp_sync_wedge.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/deprecated/rrarbiter.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/deprecated/clock_divider.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/deprecated/fifo_v2.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/deprecated/fifo_v1.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/edge_propagator_ack.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/edge_propagator.sv \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/src/edge_propagator_rx.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/.bender/git/checkouts/apb-29543f78c18455f2/src/apb_pkg.sv \
    $ROOT/.bender/git/checkouts/apb-29543f78c18455f2/src/apb_intf.sv \
    $ROOT/.bender/git/checkouts/apb-29543f78c18455f2/src/apb_err_slv.sv \
    $ROOT/.bender/git/checkouts/apb-29543f78c18455f2/src/apb_regs.sv \
    $ROOT/.bender/git/checkouts/apb-29543f78c18455f2/src/apb_cdc.sv \
    $ROOT/.bender/git/checkouts/apb-29543f78c18455f2/src/apb_demux.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_pkg.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_intf.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_atop_filter.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_burst_splitter.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_bus_compare.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_cdc_dst.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_cdc_src.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_cut.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_delayer.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_demux_simple.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_dw_downsizer.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_dw_upsizer.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_fifo.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_id_remap.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_id_prepend.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_isolate.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_join.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_lite_demux.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_lite_dw_converter.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_lite_from_mem.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_lite_join.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_lite_lfsr.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_lite_mailbox.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_lite_mux.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_lite_regs.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_lite_to_apb.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_lite_to_axi.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_modify_address.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_mux.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_rw_join.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_rw_split.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_serializer.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_slave_compare.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_throttle.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_to_detailed_mem.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_cdc.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_demux.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_err_slv.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_dw_converter.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_from_mem.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_id_serialize.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_lfsr.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_multicut.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_to_axi_lite.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_to_mem.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_zero_mem.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_interleaved_xbar.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_iw_converter.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_lite_xbar.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_xbar.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_to_mem_banked.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_to_mem_interleaved.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_to_mem_split.sv \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/src/axi_xp.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/.bender/git/checkouts/fpu_div_sqrt_mvp-021df8717282ae5e/hdl/defs_div_sqrt_mvp.sv \
    $ROOT/.bender/git/checkouts/fpu_div_sqrt_mvp-021df8717282ae5e/hdl/iteration_div_sqrt_mvp.sv \
    $ROOT/.bender/git/checkouts/fpu_div_sqrt_mvp-021df8717282ae5e/hdl/control_mvp.sv \
    $ROOT/.bender/git/checkouts/fpu_div_sqrt_mvp-021df8717282ae5e/hdl/norm_div_sqrt_mvp.sv \
    $ROOT/.bender/git/checkouts/fpu_div_sqrt_mvp-021df8717282ae5e/hdl/preprocess_mvp.sv \
    $ROOT/.bender/git/checkouts/fpu_div_sqrt_mvp-021df8717282ae5e/hdl/nrbd_nrsc_mvp.sv \
    $ROOT/.bender/git/checkouts/fpu_div_sqrt_mvp-021df8717282ae5e/hdl/div_sqrt_top_mvp.sv \
    $ROOT/.bender/git/checkouts/fpu_div_sqrt_mvp-021df8717282ae5e/hdl/div_sqrt_mvp_wrapper.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/.bender/git/checkouts/hwpe-ctrl-6a321549f37613dd/rtl/hwpe_ctrl_interfaces.sv \
    $ROOT/.bender/git/checkouts/hwpe-ctrl-6a321549f37613dd/rtl/hwpe_ctrl_package.sv \
    $ROOT/.bender/git/checkouts/hwpe-ctrl-6a321549f37613dd/rtl/hwpe_ctrl_regfile_latch.sv \
    $ROOT/.bender/git/checkouts/hwpe-ctrl-6a321549f37613dd/rtl/hwpe_ctrl_partial_mult.sv \
    $ROOT/.bender/git/checkouts/hwpe-ctrl-6a321549f37613dd/rtl/hwpe_ctrl_seq_mult.sv \
    $ROOT/.bender/git/checkouts/hwpe-ctrl-6a321549f37613dd/rtl/hwpe_ctrl_uloop.sv \
    $ROOT/.bender/git/checkouts/hwpe-ctrl-6a321549f37613dd/rtl/hwpe_ctrl_regfile_latch_test_wrap.sv \
    $ROOT/.bender/git/checkouts/hwpe-ctrl-6a321549f37613dd/rtl/hwpe_ctrl_regfile.sv \
    $ROOT/.bender/git/checkouts/hwpe-ctrl-6a321549f37613dd/rtl/hwpe_ctrl_slave.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/.bender/git/checkouts/hwpe-stream-52cf024be21233dd/rtl/hwpe_stream_interfaces.sv \
    $ROOT/.bender/git/checkouts/hwpe-stream-52cf024be21233dd/rtl/hwpe_stream_package.sv \
    $ROOT/.bender/git/checkouts/hwpe-stream-52cf024be21233dd/rtl/basic/hwpe_stream_assign.sv \
    $ROOT/.bender/git/checkouts/hwpe-stream-52cf024be21233dd/rtl/basic/hwpe_stream_buffer.sv \
    $ROOT/.bender/git/checkouts/hwpe-stream-52cf024be21233dd/rtl/basic/hwpe_stream_demux_static.sv \
    $ROOT/.bender/git/checkouts/hwpe-stream-52cf024be21233dd/rtl/basic/hwpe_stream_deserialize.sv \
    $ROOT/.bender/git/checkouts/hwpe-stream-52cf024be21233dd/rtl/basic/hwpe_stream_fence.sv \
    $ROOT/.bender/git/checkouts/hwpe-stream-52cf024be21233dd/rtl/basic/hwpe_stream_merge.sv \
    $ROOT/.bender/git/checkouts/hwpe-stream-52cf024be21233dd/rtl/basic/hwpe_stream_mux_static.sv \
    $ROOT/.bender/git/checkouts/hwpe-stream-52cf024be21233dd/rtl/basic/hwpe_stream_serialize.sv \
    $ROOT/.bender/git/checkouts/hwpe-stream-52cf024be21233dd/rtl/basic/hwpe_stream_split.sv \
    $ROOT/.bender/git/checkouts/hwpe-stream-52cf024be21233dd/rtl/fifo/hwpe_stream_fifo_ctrl.sv \
    $ROOT/.bender/git/checkouts/hwpe-stream-52cf024be21233dd/rtl/fifo/hwpe_stream_fifo_scm.sv \
    $ROOT/.bender/git/checkouts/hwpe-stream-52cf024be21233dd/rtl/streamer/hwpe_stream_addressgen.sv \
    $ROOT/.bender/git/checkouts/hwpe-stream-52cf024be21233dd/rtl/streamer/hwpe_stream_addressgen_v2.sv \
    $ROOT/.bender/git/checkouts/hwpe-stream-52cf024be21233dd/rtl/streamer/hwpe_stream_addressgen_v3.sv \
    $ROOT/.bender/git/checkouts/hwpe-stream-52cf024be21233dd/rtl/streamer/hwpe_stream_sink_realign.sv \
    $ROOT/.bender/git/checkouts/hwpe-stream-52cf024be21233dd/rtl/streamer/hwpe_stream_source_realign.sv \
    $ROOT/.bender/git/checkouts/hwpe-stream-52cf024be21233dd/rtl/streamer/hwpe_stream_strbgen.sv \
    $ROOT/.bender/git/checkouts/hwpe-stream-52cf024be21233dd/rtl/streamer/hwpe_stream_streamer_queue.sv \
    $ROOT/.bender/git/checkouts/hwpe-stream-52cf024be21233dd/rtl/tcdm/hwpe_stream_tcdm_assign.sv \
    $ROOT/.bender/git/checkouts/hwpe-stream-52cf024be21233dd/rtl/tcdm/hwpe_stream_tcdm_mux.sv \
    $ROOT/.bender/git/checkouts/hwpe-stream-52cf024be21233dd/rtl/tcdm/hwpe_stream_tcdm_mux_static.sv \
    $ROOT/.bender/git/checkouts/hwpe-stream-52cf024be21233dd/rtl/tcdm/hwpe_stream_tcdm_reorder.sv \
    $ROOT/.bender/git/checkouts/hwpe-stream-52cf024be21233dd/rtl/tcdm/hwpe_stream_tcdm_reorder_static.sv \
    $ROOT/.bender/git/checkouts/hwpe-stream-52cf024be21233dd/rtl/fifo/hwpe_stream_fifo_earlystall.sv \
    $ROOT/.bender/git/checkouts/hwpe-stream-52cf024be21233dd/rtl/fifo/hwpe_stream_fifo_earlystall_sidech.sv \
    $ROOT/.bender/git/checkouts/hwpe-stream-52cf024be21233dd/rtl/fifo/hwpe_stream_fifo_scm_test_wrap.sv \
    $ROOT/.bender/git/checkouts/hwpe-stream-52cf024be21233dd/rtl/fifo/hwpe_stream_fifo_sidech.sv \
    $ROOT/.bender/git/checkouts/hwpe-stream-52cf024be21233dd/rtl/fifo/hwpe_stream_fifo.sv \
    $ROOT/.bender/git/checkouts/hwpe-stream-52cf024be21233dd/rtl/tcdm/hwpe_stream_tcdm_fifo_load_sidech.sv \
    $ROOT/.bender/git/checkouts/hwpe-stream-52cf024be21233dd/rtl/fifo/hwpe_stream_fifo_passthrough.sv \
    $ROOT/.bender/git/checkouts/hwpe-stream-52cf024be21233dd/rtl/streamer/hwpe_stream_source.sv \
    $ROOT/.bender/git/checkouts/hwpe-stream-52cf024be21233dd/rtl/tcdm/hwpe_stream_tcdm_fifo.sv \
    $ROOT/.bender/git/checkouts/hwpe-stream-52cf024be21233dd/rtl/tcdm/hwpe_stream_tcdm_fifo_load.sv \
    $ROOT/.bender/git/checkouts/hwpe-stream-52cf024be21233dd/rtl/tcdm/hwpe_stream_tcdm_fifo_store.sv \
    $ROOT/.bender/git/checkouts/hwpe-stream-52cf024be21233dd/rtl/streamer/hwpe_stream_sink.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/.bender/git/checkouts/axi_riscv_atomics-c098c3f53f43f41f/src/axi_res_tbl.sv \
    $ROOT/.bender/git/checkouts/axi_riscv_atomics-c098c3f53f43f41f/src/axi_riscv_amos_alu.sv \
    $ROOT/.bender/git/checkouts/axi_riscv_atomics-c098c3f53f43f41f/src/axi_riscv_amos.sv \
    $ROOT/.bender/git/checkouts/axi_riscv_atomics-c098c3f53f43f41f/src/axi_riscv_lrsc.sv \
    $ROOT/.bender/git/checkouts/axi_riscv_atomics-c098c3f53f43f41f/src/axi_riscv_atomics.sv \
    $ROOT/.bender/git/checkouts/axi_riscv_atomics-c098c3f53f43f41f/src/axi_riscv_lrsc_wrap.sv \
    $ROOT/.bender/git/checkouts/axi_riscv_atomics-c098c3f53f43f41f/src/axi_riscv_amos_wrap.sv \
    $ROOT/.bender/git/checkouts/axi_riscv_atomics-c098c3f53f43f41f/src/axi_riscv_atomics_wrap.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/.bender/git/checkouts/fpnew-d366da075c7a414c/src/fpnew_pkg.sv \
    $ROOT/.bender/git/checkouts/fpnew-d366da075c7a414c/src/fpnew_cast_multi.sv \
    $ROOT/.bender/git/checkouts/fpnew-d366da075c7a414c/src/fpnew_classifier.sv \
    $ROOT/.bender/git/checkouts/fpnew-d366da075c7a414c/vendor/opene906/E906_RTL_FACTORY/gen_rtl/clk/rtl/gated_clk_cell.v \
    $ROOT/.bender/git/checkouts/fpnew-d366da075c7a414c/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_ctrl.v \
    $ROOT/.bender/git/checkouts/fpnew-d366da075c7a414c/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_ff1.v \
    $ROOT/.bender/git/checkouts/fpnew-d366da075c7a414c/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_pack_single.v \
    $ROOT/.bender/git/checkouts/fpnew-d366da075c7a414c/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_prepare.v \
    $ROOT/.bender/git/checkouts/fpnew-d366da075c7a414c/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_round_single.v \
    $ROOT/.bender/git/checkouts/fpnew-d366da075c7a414c/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_special.v \
    $ROOT/.bender/git/checkouts/fpnew-d366da075c7a414c/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_srt_single.v \
    $ROOT/.bender/git/checkouts/fpnew-d366da075c7a414c/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_top.v \
    $ROOT/.bender/git/checkouts/fpnew-d366da075c7a414c/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fpu/rtl/pa_fpu_dp.v \
    $ROOT/.bender/git/checkouts/fpnew-d366da075c7a414c/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fpu/rtl/pa_fpu_frbus.v \
    $ROOT/.bender/git/checkouts/fpnew-d366da075c7a414c/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fpu/rtl/pa_fpu_src_type.v \
    $ROOT/.bender/git/checkouts/fpnew-d366da075c7a414c/src/fpnew_divsqrt_th_32.sv \
    $ROOT/.bender/git/checkouts/fpnew-d366da075c7a414c/src/fpnew_divsqrt_multi.sv \
    $ROOT/.bender/git/checkouts/fpnew-d366da075c7a414c/src/fpnew_fma.sv \
    $ROOT/.bender/git/checkouts/fpnew-d366da075c7a414c/src/fpnew_fma_multi.sv \
    $ROOT/.bender/git/checkouts/fpnew-d366da075c7a414c/src/fpnew_sdotp_multi.sv \
    $ROOT/.bender/git/checkouts/fpnew-d366da075c7a414c/src/fpnew_sdotp_multi_wrapper.sv \
    $ROOT/.bender/git/checkouts/fpnew-d366da075c7a414c/src/fpnew_noncomp.sv \
    $ROOT/.bender/git/checkouts/fpnew-d366da075c7a414c/src/fpnew_opgroup_block.sv \
    $ROOT/.bender/git/checkouts/fpnew-d366da075c7a414c/src/fpnew_opgroup_fmt_slice.sv \
    $ROOT/.bender/git/checkouts/fpnew-d366da075c7a414c/src/fpnew_opgroup_multifmt_slice.sv \
    $ROOT/.bender/git/checkouts/fpnew-d366da075c7a414c/src/fpnew_rounding.sv \
    $ROOT/.bender/git/checkouts/fpnew-d366da075c7a414c/src/lfsr_sr.sv \
    $ROOT/.bender/git/checkouts/fpnew-d366da075c7a414c/src/fpnew_top.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/.bender/git/checkouts/hwpe-mac-engine-c5f1d60119de571e/rtl/mac_package.sv \
    $ROOT/.bender/git/checkouts/hwpe-mac-engine-c5f1d60119de571e/rtl/mac_engine.sv \
    $ROOT/.bender/git/checkouts/hwpe-mac-engine-c5f1d60119de571e/rtl/mac_fsm.sv \
    $ROOT/.bender/git/checkouts/hwpe-mac-engine-c5f1d60119de571e/rtl/mac_streamer.sv \
    $ROOT/.bender/git/checkouts/hwpe-mac-engine-c5f1d60119de571e/rtl/mac_ctrl.sv \
    $ROOT/.bender/git/checkouts/hwpe-mac-engine-c5f1d60119de571e/rtl/mac_top.sv \
    $ROOT/.bender/git/checkouts/hwpe-mac-engine-c5f1d60119de571e/wrap/mac_top_wrap.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/.bender/git/checkouts/register_interface-257051d7114a985f/src/reg_intf.sv \
    $ROOT/.bender/git/checkouts/register_interface-257051d7114a985f/vendor/lowrisc_opentitan/src/prim_subreg_arb.sv \
    $ROOT/.bender/git/checkouts/register_interface-257051d7114a985f/vendor/lowrisc_opentitan/src/prim_subreg_ext.sv \
    $ROOT/.bender/git/checkouts/register_interface-257051d7114a985f/src/apb_to_reg.sv \
    $ROOT/.bender/git/checkouts/register_interface-257051d7114a985f/src/axi_lite_to_reg.sv \
    $ROOT/.bender/git/checkouts/register_interface-257051d7114a985f/src/axi_to_reg_v2.sv \
    $ROOT/.bender/git/checkouts/register_interface-257051d7114a985f/src/periph_to_reg.sv \
    $ROOT/.bender/git/checkouts/register_interface-257051d7114a985f/src/reg_cdc.sv \
    $ROOT/.bender/git/checkouts/register_interface-257051d7114a985f/src/reg_cut.sv \
    $ROOT/.bender/git/checkouts/register_interface-257051d7114a985f/src/reg_demux.sv \
    $ROOT/.bender/git/checkouts/register_interface-257051d7114a985f/src/reg_err_slv.sv \
    $ROOT/.bender/git/checkouts/register_interface-257051d7114a985f/src/reg_filter_empty_writes.sv \
    $ROOT/.bender/git/checkouts/register_interface-257051d7114a985f/src/reg_mux.sv \
    $ROOT/.bender/git/checkouts/register_interface-257051d7114a985f/src/reg_to_apb.sv \
    $ROOT/.bender/git/checkouts/register_interface-257051d7114a985f/src/reg_to_mem.sv \
    $ROOT/.bender/git/checkouts/register_interface-257051d7114a985f/src/reg_to_tlul.sv \
    $ROOT/.bender/git/checkouts/register_interface-257051d7114a985f/src/reg_to_axi.sv \
    $ROOT/.bender/git/checkouts/register_interface-257051d7114a985f/src/reg_uniform.sv \
    $ROOT/.bender/git/checkouts/register_interface-257051d7114a985f/vendor/lowrisc_opentitan/src/prim_subreg_shadow.sv \
    $ROOT/.bender/git/checkouts/register_interface-257051d7114a985f/vendor/lowrisc_opentitan/src/prim_subreg.sv \
    $ROOT/.bender/git/checkouts/register_interface-257051d7114a985f/src/deprecated/axi_to_reg.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/.bender/git/checkouts/riscv-dbg-0d4710fc93fcfe2a/src/dm_pkg.sv \
    $ROOT/.bender/git/checkouts/riscv-dbg-0d4710fc93fcfe2a/debug_rom/debug_rom.sv \
    $ROOT/.bender/git/checkouts/riscv-dbg-0d4710fc93fcfe2a/debug_rom/debug_rom_one_scratch.sv \
    $ROOT/.bender/git/checkouts/riscv-dbg-0d4710fc93fcfe2a/src/dm_csrs.sv \
    $ROOT/.bender/git/checkouts/riscv-dbg-0d4710fc93fcfe2a/src/dm_mem.sv \
    $ROOT/.bender/git/checkouts/riscv-dbg-0d4710fc93fcfe2a/src/dmi_cdc.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/.bender/git/checkouts/riscv-dbg-0d4710fc93fcfe2a/src/dmi_jtag_tap.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/.bender/git/checkouts/riscv-dbg-0d4710fc93fcfe2a/src/dm_sba.sv \
    $ROOT/.bender/git/checkouts/riscv-dbg-0d4710fc93fcfe2a/src/dm_top.sv \
    $ROOT/.bender/git/checkouts/riscv-dbg-0d4710fc93fcfe2a/src/dmi_jtag.sv \
    $ROOT/.bender/git/checkouts/riscv-dbg-0d4710fc93fcfe2a/src/dm_obi_top.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/.bender/git/checkouts/scm-282acb124ce4b412/fpga_scm/register_file_1r_1w_all.sv \
    $ROOT/.bender/git/checkouts/scm-282acb124ce4b412/fpga_scm/register_file_1r_1w_be.sv \
    $ROOT/.bender/git/checkouts/scm-282acb124ce4b412/fpga_scm/register_file_1r_1w.sv \
    $ROOT/.bender/git/checkouts/scm-282acb124ce4b412/fpga_scm/register_file_1r_1w_1row.sv \
    $ROOT/.bender/git/checkouts/scm-282acb124ce4b412/fpga_scm/register_file_1r_1w_raw.sv \
    $ROOT/.bender/git/checkouts/scm-282acb124ce4b412/fpga_scm/register_file_1w_multi_port_read.sv \
    $ROOT/.bender/git/checkouts/scm-282acb124ce4b412/fpga_scm/register_file_1w_64b_multi_port_read_32b.sv \
    $ROOT/.bender/git/checkouts/scm-282acb124ce4b412/fpga_scm/register_file_1w_64b_1r_32b.sv \
    $ROOT/.bender/git/checkouts/scm-282acb124ce4b412/fpga_scm/register_file_2r_1w_asymm.sv \
    $ROOT/.bender/git/checkouts/scm-282acb124ce4b412/fpga_scm/register_file_2r_1w_asymm_test_wrap.sv \
    $ROOT/.bender/git/checkouts/scm-282acb124ce4b412/fpga_scm/register_file_2r_2w.sv \
    $ROOT/.bender/git/checkouts/scm-282acb124ce4b412/fpga_scm/register_file_3r_2w.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/.bender/git/checkouts/apb_timer-1973c87a95c2cf35/src/timer.sv \
    $ROOT/.bender/git/checkouts/apb_timer-1973c87a95c2cf35/src/apb_timer.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/.bender/git/checkouts/apb_uart-3d80c89a89259e46/src/slib_clock_div.sv \
    $ROOT/.bender/git/checkouts/apb_uart-3d80c89a89259e46/src/slib_counter.sv \
    $ROOT/.bender/git/checkouts/apb_uart-3d80c89a89259e46/src/slib_edge_detect.sv \
    $ROOT/.bender/git/checkouts/apb_uart-3d80c89a89259e46/src/slib_fifo.sv \
    $ROOT/.bender/git/checkouts/apb_uart-3d80c89a89259e46/src/slib_input_filter.sv \
    $ROOT/.bender/git/checkouts/apb_uart-3d80c89a89259e46/src/slib_input_sync.sv \
    $ROOT/.bender/git/checkouts/apb_uart-3d80c89a89259e46/src/slib_mv_filter.sv \
    $ROOT/.bender/git/checkouts/apb_uart-3d80c89a89259e46/src/uart_baudgen.sv \
    $ROOT/.bender/git/checkouts/apb_uart-3d80c89a89259e46/src/uart_interrupt.sv \
    $ROOT/.bender/git/checkouts/apb_uart-3d80c89a89259e46/src/uart_receiver.sv \
    $ROOT/.bender/git/checkouts/apb_uart-3d80c89a89259e46/src/uart_transmitter.sv \
    $ROOT/.bender/git/checkouts/apb_uart-3d80c89a89259e46/src/apb_uart.sv \
    $ROOT/.bender/git/checkouts/apb_uart-3d80c89a89259e46/src/apb_uart_wrap.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/hw/vendor/pulp_platform_axi_tlb/src/axi_tlb_l1_chan.sv \
    $ROOT/hw/vendor/pulp_platform_axi_tlb/src/axi_tlb_reg_pkg.sv \
    $ROOT/hw/vendor/pulp_platform_axi_tlb/src/axi_tlb_reg_top.sv \
    $ROOT/hw/vendor/pulp_platform_axi_tlb/src/axi_tlb_l1.sv \
    $ROOT/hw/vendor/pulp_platform_axi_tlb/src/axi_tlb_noreg.sv \
    $ROOT/hw/vendor/pulp_platform_axi_tlb/src/axi_tlb.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/.bender/git/checkouts/clint-89bd4278406fcf7f/src/clint_reg_pkg.sv \
    $ROOT/.bender/git/checkouts/clint-89bd4278406fcf7f/src/clint_reg_top.sv \
    $ROOT/.bender/git/checkouts/clint-89bd4278406fcf7f/src/clint.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/.bender/git/checkouts/cluster_icache-b8ee03c23594c472/src/snitch_icache_pkg.sv \
    $ROOT/.bender/git/checkouts/cluster_icache-b8ee03c23594c472/src/riscv_instr_branch.sv \
    $ROOT/.bender/git/checkouts/cluster_icache-b8ee03c23594c472/src/snitch_axi_to_cache.sv \
    $ROOT/.bender/git/checkouts/cluster_icache-b8ee03c23594c472/src/snitch_icache_l0.sv \
    $ROOT/.bender/git/checkouts/cluster_icache-b8ee03c23594c472/src/snitch_icache_refill.sv \
    $ROOT/.bender/git/checkouts/cluster_icache-b8ee03c23594c472/src/snitch_icache_lfsr.sv \
    $ROOT/.bender/git/checkouts/cluster_icache-b8ee03c23594c472/src/snitch_icache_lookup_parallel.sv \
    $ROOT/.bender/git/checkouts/cluster_icache-b8ee03c23594c472/src/snitch_icache_lookup_serial.sv \
    $ROOT/.bender/git/checkouts/cluster_icache-b8ee03c23594c472/src/snitch_icache_handler.sv \
    $ROOT/.bender/git/checkouts/cluster_icache-b8ee03c23594c472/src/snitch_icache.sv \
    $ROOT/.bender/git/checkouts/cluster_icache-b8ee03c23594c472/src/snitch_read_only_cache.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/hw/vendor/openhwgroup_cva6/core/include/cv64a6_imafdc_sv39_config_pkg.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/include/riscv_pkg.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/include/ariane_dm_pkg.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/include/ariane_pkg.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/mmu_sv39/tlb.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/mmu_sv39/mmu.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/mmu_sv39/ptw.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/hw/vendor/openhwgroup_cva6/core/include/ariane_rvfi_pkg.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/include/ariane_axi_pkg.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/include/wt_cache_pkg.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/include/std_cache_pkg.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/include/axi_intf.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/hw/vendor/openhwgroup_cva6/vendor/pulp-platform/fpga-support/rtl/SyncDpRam.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/vendor/pulp-platform/fpga-support/rtl/AsyncDpRam.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/vendor/pulp-platform/fpga-support/rtl/AsyncThreePortRam.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/hw/vendor/openhwgroup_cva6/core/include/cvxif_pkg.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/cvxif_example/include/cvxif_instr_pkg.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/cvxif_fu.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/cvxif_example/cvxif_example_coprocessor.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/cvxif_example/instr_decoder.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/ariane.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/cva6.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/alu.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/fpu_wrap.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/branch_unit.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/compressed_decoder.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/controller.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/csr_buffer.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/csr_regfile.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/decoder.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/ex_stage.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/instr_realign.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/id_stage.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/issue_read_operands.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/issue_stage.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/load_unit.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/load_store_unit.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/lsu_bypass.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/mult.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/multiplier.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/serdiv.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/perf_counters.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/ariane_regfile_ff.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/ariane_regfile_fpga.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/re_name.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/scoreboard.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/store_buffer.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/amo_buffer.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/store_unit.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/commit_stage.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/axi_shim.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/frontend/btb.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/frontend/bht.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/frontend/ras.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/frontend/instr_scan.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/frontend/instr_queue.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/frontend/frontend.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/cache_subsystem/wt_dcache_ctrl.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/cache_subsystem/wt_dcache_mem.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/cache_subsystem/wt_dcache_missunit.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/cache_subsystem/wt_dcache_wbuffer.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/cache_subsystem/wt_dcache.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/cache_subsystem/cva6_icache.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/cache_subsystem/wt_cache_subsystem.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/cache_subsystem/wt_axi_adapter.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/cache_subsystem/tag_cmp.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/cache_subsystem/cache_ctrl.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/cache_subsystem/amo_alu.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/cache_subsystem/wt_l15_adapter.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/corev_apu/tb/axi_adapter.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/cache_subsystem/miss_handler.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/cache_subsystem/std_nbdcache.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/cache_subsystem/cva6_icache_axi_wrapper.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/cache_subsystem/std_cache_subsystem.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/pmp/src/pmp.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/core/pmp/src/pmp_entry.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/hw/vendor/openhwgroup_cva6/common/local/util/sram.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/hw/vendor/openhwgroup_cva6/common/local/util/tc_sram_fpga_wrapper.sv \
    $ROOT/hw/vendor/openhwgroup_cva6/vendor/pulp-platform/fpga-support/rtl/SyncSpRamBeNx64.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/prim/prim_pulp_platform/prim_flop_2sync.sv \
    $ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/prim/rtl/prim_util_pkg.sv \
    $ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/prim/rtl/prim_sync_reqack.sv \
    $ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/prim/rtl/prim_sync_reqack_data.sv \
    $ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/prim/rtl/prim_pulse_sync.sv \
    $ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/prim/rtl/prim_packer_fifo.sv \
    $ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/prim/rtl/prim_fifo_sync.sv \
    $ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/prim/rtl/prim_filter_ctr.sv \
    $ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/prim/rtl/prim_intr_hw.sv \
    $ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/prim/rtl/prim_fifo_async.sv \
    $ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/gpio/rtl/gpio_reg_pkg.sv \
    $ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/i2c/rtl/i2c_reg_pkg.sv \
    $ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/spi_host/rtl/spi_host_reg_pkg.sv \
    $ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/gpio/rtl/gpio_reg_top.sv \
    $ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/i2c/rtl/i2c_reg_top.sv \
    $ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/spi_host/rtl/spi_host_reg_top.sv \
    $ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/i2c/rtl/i2c_fsm.sv \
    $ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/rv_plic/rtl/rv_plic_gateway.sv \
    $ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/spi_host/rtl/spi_host_byte_merge.sv \
    $ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/spi_host/rtl/spi_host_byte_select.sv \
    $ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/spi_host/rtl/spi_host_cmd_pkg.sv \
    $ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/spi_host/rtl/spi_host_command_cdc.sv \
    $ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/spi_host/rtl/spi_host_fsm.sv \
    $ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/spi_host/rtl/spi_host_window.sv \
    $ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/spi_host/rtl/spi_host_data_cdc.sv \
    $ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/spi_host/rtl/spi_host_shift_register.sv \
    $ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/i2c/rtl/i2c_core.sv \
    $ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/rv_plic/rtl/rv_plic_target.sv \
    $ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/spi_host/rtl/spi_host_core.sv \
    $ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/gpio/rtl/gpio.sv \
    $ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/i2c/rtl/i2c.sv \
    $ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/spi_host/rtl/spi_host.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/future/src/mem_to_axi_lite.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/future/src/idma_reg64_frontend_reg_pkg.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/future/src/idma_tf_id_gen.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/future/src/dma/axi_dma_data_path.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/future/src/axi_interleaved_xbar.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/future/src/axi_zero_mem.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/future/src/idma_reg64_frontend_reg_top.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/future/src/idma_reg64_frontend.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/future/src/dma/axi_dma_data_mover.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/future/src/dma/axi_dma_burst_reshaper.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/future/src/dma/axi_dma_backend.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/reqrsp_interface/src/reqrsp_pkg.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/reqrsp_interface/src/reqrsp_intf.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/reqrsp_interface/src/axi_to_reqrsp.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/reqrsp_interface/src/reqrsp_cut.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/reqrsp_interface/src/reqrsp_demux.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/reqrsp_interface/src/reqrsp_iso.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/reqrsp_interface/src/reqrsp_mux.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/reqrsp_interface/src/reqrsp_to_axi.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/mem_interface/src/mem_wide_narrow_mux.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/mem_interface/src/mem_interface.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/tcdm_interface/src/tcdm_interface.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/tcdm_interface/src/axi_to_tcdm.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/tcdm_interface/src/reqrsp_to_tcdm.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/tcdm_interface/src/tcdm_mux.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch/src/snitch_pma_pkg.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch/src/riscv_instr.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch/src/csr_snax_def.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch/src/snitch_pkg.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch/src/snitch_regfile_ff.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch/src/snitch_lsu.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch/src/snitch_l0_tlb.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch/src/snitch.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snax_hwpe_mac/src/snax_hwpe_ctrl.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snax_hwpe_mac/src/snax_hwpe_to_reqrsp.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snax_hwpe_mac/src/snax_mac_wrapper.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch_vm/src/snitch_ptw.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch_dma/src/axi_dma_pkg.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch_dma/src/axi_dma_error_handler.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch_dma/src/axi_dma_perf_counters.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch_dma/src/axi_dma_twod_ext.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch_dma/src/axi_dma_tc_snitch_fe.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch_icache/src/snitch_icache_pkg.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch_icache/src/snitch_icache_l0.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch_icache/src/snitch_icache_refill.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch_icache/src/snitch_icache_lfsr.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch_icache/src/snitch_icache_lookup.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch_icache/src/snitch_icache_handler.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch_icache/src/snitch_icache.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch_ipu/src/snitch_ipu_pkg.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch_ipu/src/snitch_ipu_alu.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch_ipu/src/snitch_int_ss.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch_ssr/src/snitch_ssr_pkg.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch_ssr/src/snitch_ssr_switch.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch_ssr/src/snitch_ssr_credit_counter.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch_ssr/src/snitch_ssr_indirector.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch_ssr/src/snitch_ssr_intersector.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch_ssr/src/snitch_ssr_addr_gen.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch_ssr/src/snitch_ssr.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch_ssr/src/snitch_ssr_streamer.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch_cluster/src/snitch_amo_shim.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch_cluster/src/snitch_cluster_peripheral/snitch_cluster_peripheral_reg_pkg.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch_cluster/src/snitch_cluster_peripheral/snitch_cluster_peripheral_reg_top.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch_cluster/src/snitch_cluster_peripheral/snitch_cluster_peripheral.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch_cluster/src/snitch_fpu.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch_cluster/src/snitch_sequencer.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch_cluster/src/snitch_tcdm_interconnect.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch_cluster/src/snitch_barrier.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch_cluster/src/snitch_fp_ss.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch_cluster/src/snitch_shared_muldiv.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch_cluster/src/snitch_cc.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch_cluster/src/snitch_clkdiv2.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch_cluster/src/snitch_hive.sv \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch_cluster/src/snitch_cluster.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snax/src/snax_acc_mux_demux.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/hw/future/reg_to_apb.sv \
    $ROOT/hw/snitch_read_only_cache/src/snitch_axi_to_cache.sv \
    $ROOT/hw/snitch_read_only_cache/src/snitch_read_only_cache.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/hw/spm_interface/src/spm_interface.sv \
    $ROOT/hw/spm_interface/src/spm_rmw_adapter.sv \
    $ROOT/hw/spm_interface/src/spm_1p_adv.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/hw/occamy/soc_ctrl/occamy_soc_ctrl.sv \
]
add_files -norecurse -fileset [current_fileset] [list \
    $ROOT/target/sim/src/occamy_xilinx.sv \
]

set_property include_dirs [list \
    $ROOT/.bender/git/checkouts/apb-29543f78c18455f2/include \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/include \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/include \
    $ROOT/.bender/git/checkouts/hwpe-ctrl-6a321549f37613dd/rtl \
    $ROOT/.bender/git/checkouts/hwpe-mac-engine-c5f1d60119de571e/rtl \
    $ROOT/.bender/git/checkouts/hwpe-stream-52cf024be21233dd/rtl \
    $ROOT/.bender/git/checkouts/register_interface-257051d7114a985f/include \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/mem_interface/include \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/reqrsp_interface/include \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch/include \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch_ssr/include \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/tcdm_interface/include \
    $ROOT/hw/spm_interface/include \
    $ROOT/hw/vendor/openhwgroup_cva6/common/local/util \
    $ROOT/hw/vendor/pulp_platform_axi_tlb/include \
] [current_fileset]

set_property include_dirs [list \
    $ROOT/.bender/git/checkouts/apb-29543f78c18455f2/include \
    $ROOT/.bender/git/checkouts/axi-7a452571dc317362/include \
    $ROOT/.bender/git/checkouts/common_cells-e1f1589900f5e5d2/include \
    $ROOT/.bender/git/checkouts/hwpe-ctrl-6a321549f37613dd/rtl \
    $ROOT/.bender/git/checkouts/hwpe-mac-engine-c5f1d60119de571e/rtl \
    $ROOT/.bender/git/checkouts/hwpe-stream-52cf024be21233dd/rtl \
    $ROOT/.bender/git/checkouts/register_interface-257051d7114a985f/include \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/mem_interface/include \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/reqrsp_interface/include \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch/include \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/snitch_ssr/include \
    $ROOT/.bender/git/checkouts/snitch_cluster-4cb275b8763a2b0d/hw/tcdm_interface/include \
    $ROOT/hw/spm_interface/include \
    $ROOT/hw/vendor/openhwgroup_cva6/common/local/util \
    $ROOT/hw/vendor/pulp_platform_axi_tlb/include \
] [current_fileset -simset]

set_property verilog_define [list \
    SNITCH_ENABLE_PERF \
    TARGET_CV64A6_IMAFDC_SV39 \
    TARGET_FPGA \
    TARGET_SYNTHESIS \
    TARGET_VIVADO \
    TARGET_XILINX \
    WB_DCACHE=1 \
] [current_fileset]

set_property verilog_define [list \
    SNITCH_ENABLE_PERF \
    TARGET_CV64A6_IMAFDC_SV39 \
    TARGET_FPGA \
    TARGET_SYNTHESIS \
    TARGET_VIVADO \
    TARGET_XILINX \
    WB_DCACHE=1 \
] [current_fileset -simset]
