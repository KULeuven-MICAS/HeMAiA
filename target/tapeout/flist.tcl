# This script was generated automatically by bender.
set ROOT "/users/micas/ydeng/Documents/HeMAiA_Synthesis/HeMAiA"
set search_path_initial $search_path

set search_path $search_path_initial

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/hw/hemaia/tech_cells_tsmc16/src/tsmc16/tc_sram_tsmc16.sv" \
        "$ROOT/hw/hemaia/tech_cells_tsmc16/src/rtl/tc_sram_impl.sv" \
    ]
]} {return 1}

set search_path $search_path_initial

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/hw/hemaia/tech_cells_tsmc16/src/rtl/tc_clk.sv" \
    ]
]} {return 1}

set search_path $search_path_initial

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/hw/hemaia/tech_cells_tsmc16/src/deprecated/pulp_clock_gating_async.sv" \
        "$ROOT/hw/hemaia/tech_cells_tsmc16/src/deprecated/cluster_clk_cells.sv" \
        "$ROOT/hw/hemaia/tech_cells_tsmc16/src/deprecated/pulp_clk_cells.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/binary_to_gray.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/cb_filter_pkg.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/cc_onehot.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/cdc_reset_ctrlr_pkg.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/cf_math_pkg.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/clk_int_div.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/credit_counter.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/delta_counter.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/ecc_pkg.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/edge_propagator_tx.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/exp_backoff.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/fifo_v3.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/gray_to_binary.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/isochronous_4phase_handshake.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/isochronous_spill_register.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/lfsr.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/lfsr_16bit.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/lfsr_8bit.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/lossy_valid_to_stream.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/mv_filter.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/onehot_to_bin.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/plru_tree.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/passthrough_stream_fifo.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/popcount.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/rr_arb_tree.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/rstgen_bypass.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/serial_deglitch.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/shift_reg.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/shift_reg_gated.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/spill_register_flushable.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/stream_demux.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/stream_filter.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/stream_fork.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/stream_intf.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/stream_join_dynamic.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/stream_mux.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/stream_throttle.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/sub_per_hash.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/sync.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/sync_wedge.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/unread.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/read.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/addr_decode_dync.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/cdc_2phase.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/cdc_4phase.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/clk_int_div_static.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/addr_decode.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/addr_decode_napot.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/multiaddr_decode.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/cb_filter.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/cdc_fifo_2phase.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/clk_mux_glitch_free.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/counter.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/ecc_decode.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/ecc_encode.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/edge_detect.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/lzc.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/max_counter.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/rstgen.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/spill_register.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/stream_delay.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/stream_fifo.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/stream_fork_dynamic.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/stream_join.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/cdc_reset_ctrlr.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/cdc_fifo_gray.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/fall_through_register.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/id_queue.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/stream_to_mem.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/stream_arbiter_flushable.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/stream_fifo_optimal_wrap.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/stream_register.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/stream_xbar.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/cdc_fifo_gray_clearable.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/cdc_2phase_clearable.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/mem_to_banks_detailed.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/stream_arbiter.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/stream_omega_net.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/mem_to_banks.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/deprecated/clock_divider_counter.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/deprecated/clk_div.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/deprecated/find_first_one.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/deprecated/generic_LFSR_8bit.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/deprecated/generic_fifo.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/deprecated/prioarbiter.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/deprecated/pulp_sync.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/deprecated/pulp_sync_wedge.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/deprecated/rrarbiter.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/deprecated/clock_divider.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/deprecated/fifo_v2.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/deprecated/fifo_v1.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/edge_propagator_ack.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/edge_propagator.sv" \
        "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/src/edge_propagator_rx.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/apb-967c1a20a8f82c97/include"
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/apb-967c1a20a8f82c97/src/apb_pkg.sv" \
        "$ROOT/.bender/git/checkouts/apb-967c1a20a8f82c97/src/apb_intf.sv" \
        "$ROOT/.bender/git/checkouts/apb-967c1a20a8f82c97/src/apb_err_slv.sv" \
        "$ROOT/.bender/git/checkouts/apb-967c1a20a8f82c97/src/apb_regs.sv" \
        "$ROOT/.bender/git/checkouts/apb-967c1a20a8f82c97/src/apb_cdc.sv" \
        "$ROOT/.bender/git/checkouts/apb-967c1a20a8f82c97/src/apb_demux.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"
lappend search_path "$ROOT/hw/vendor/pulp_platform_axi/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_pkg.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_intf.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_atop_filter.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_burst_splitter.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_bus_compare.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_cdc_dst.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_cdc_src.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_cut.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_delayer.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_demux_simple.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_dw_downsizer.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_dw_upsizer.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_fifo.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_id_remap.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_id_prepend.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_isolate.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_join.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_lite_demux.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_lite_dw_converter.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_lite_from_mem.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_lite_join.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_lite_lfsr.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_lite_mailbox.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_lite_mux.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_lite_regs.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_lite_to_apb.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_lite_to_axi.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_modify_address.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_mux.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_rw_join.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_rw_split.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_serializer.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_slave_compare.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_throttle.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_to_detailed_mem.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_cdc.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_demux.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_err_slv.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_dw_converter.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_from_mem.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_id_serialize.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_lfsr.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_multicut.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_to_axi_lite.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_to_mem.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_iw_converter.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_lite_xbar.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_xbar.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_to_mem_banked.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_to_mem_interleaved.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_to_mem_split.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi/src/axi_xp.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/axi_stream-683bf88f199f0ed8/include"
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/axi_stream-683bf88f199f0ed8/src/axi_stream_intf.sv" \
        "$ROOT/.bender/git/checkouts/axi_stream-683bf88f199f0ed8/src/axi_stream_cut.sv" \
        "$ROOT/.bender/git/checkouts/axi_stream-683bf88f199f0ed8/src/axi_stream_dw_downsizer.sv" \
        "$ROOT/.bender/git/checkouts/axi_stream-683bf88f199f0ed8/src/axi_stream_dw_upsizer.sv" \
        "$ROOT/.bender/git/checkouts/axi_stream-683bf88f199f0ed8/src/axi_stream_multicut.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/fpu_div_sqrt_mvp-dda3f2b23b1e3cf9/hdl/defs_div_sqrt_mvp.sv" \
        "$ROOT/.bender/git/checkouts/fpu_div_sqrt_mvp-dda3f2b23b1e3cf9/hdl/iteration_div_sqrt_mvp.sv" \
        "$ROOT/.bender/git/checkouts/fpu_div_sqrt_mvp-dda3f2b23b1e3cf9/hdl/control_mvp.sv" \
        "$ROOT/.bender/git/checkouts/fpu_div_sqrt_mvp-dda3f2b23b1e3cf9/hdl/norm_div_sqrt_mvp.sv" \
        "$ROOT/.bender/git/checkouts/fpu_div_sqrt_mvp-dda3f2b23b1e3cf9/hdl/preprocess_mvp.sv" \
        "$ROOT/.bender/git/checkouts/fpu_div_sqrt_mvp-dda3f2b23b1e3cf9/hdl/nrbd_nrsc_mvp.sv" \
        "$ROOT/.bender/git/checkouts/fpu_div_sqrt_mvp-dda3f2b23b1e3cf9/hdl/div_sqrt_top_mvp.sv" \
        "$ROOT/.bender/git/checkouts/fpu_div_sqrt_mvp-dda3f2b23b1e3cf9/hdl/div_sqrt_mvp_wrapper.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/hwpe-ctrl-33741c03c89e9bd2/rtl"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/hwpe-ctrl-33741c03c89e9bd2/rtl/hwpe_ctrl_interfaces.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-ctrl-33741c03c89e9bd2/rtl/hwpe_ctrl_package.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-ctrl-33741c03c89e9bd2/rtl/hwpe_ctrl_regfile_latch.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-ctrl-33741c03c89e9bd2/rtl/hwpe_ctrl_partial_mult.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-ctrl-33741c03c89e9bd2/rtl/hwpe_ctrl_seq_mult.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-ctrl-33741c03c89e9bd2/rtl/hwpe_ctrl_uloop.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-ctrl-33741c03c89e9bd2/rtl/hwpe_ctrl_regfile_latch_test_wrap.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-ctrl-33741c03c89e9bd2/rtl/hwpe_ctrl_regfile.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-ctrl-33741c03c89e9bd2/rtl/hwpe_ctrl_slave.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/hwpe-stream-c69cd5e9f8a2202f/rtl"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/hwpe-stream-c69cd5e9f8a2202f/rtl/hwpe_stream_package.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-c69cd5e9f8a2202f/rtl/hwpe_stream_interfaces.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-c69cd5e9f8a2202f/rtl/basic/hwpe_stream_assign.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-c69cd5e9f8a2202f/rtl/basic/hwpe_stream_buffer.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-c69cd5e9f8a2202f/rtl/basic/hwpe_stream_demux_static.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-c69cd5e9f8a2202f/rtl/basic/hwpe_stream_deserialize.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-c69cd5e9f8a2202f/rtl/basic/hwpe_stream_fence.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-c69cd5e9f8a2202f/rtl/basic/hwpe_stream_merge.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-c69cd5e9f8a2202f/rtl/basic/hwpe_stream_mux_static.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-c69cd5e9f8a2202f/rtl/basic/hwpe_stream_serialize.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-c69cd5e9f8a2202f/rtl/basic/hwpe_stream_split.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-c69cd5e9f8a2202f/rtl/fifo/hwpe_stream_fifo_ctrl.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-c69cd5e9f8a2202f/rtl/fifo/hwpe_stream_fifo_scm.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-c69cd5e9f8a2202f/rtl/streamer/hwpe_stream_addressgen.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-c69cd5e9f8a2202f/rtl/streamer/hwpe_stream_addressgen_v2.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-c69cd5e9f8a2202f/rtl/streamer/hwpe_stream_addressgen_v3.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-c69cd5e9f8a2202f/rtl/streamer/hwpe_stream_sink_realign.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-c69cd5e9f8a2202f/rtl/streamer/hwpe_stream_source_realign.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-c69cd5e9f8a2202f/rtl/streamer/hwpe_stream_strbgen.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-c69cd5e9f8a2202f/rtl/streamer/hwpe_stream_streamer_queue.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-c69cd5e9f8a2202f/rtl/tcdm/hwpe_stream_tcdm_assign.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-c69cd5e9f8a2202f/rtl/tcdm/hwpe_stream_tcdm_mux.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-c69cd5e9f8a2202f/rtl/tcdm/hwpe_stream_tcdm_mux_static.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-c69cd5e9f8a2202f/rtl/tcdm/hwpe_stream_tcdm_reorder.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-c69cd5e9f8a2202f/rtl/tcdm/hwpe_stream_tcdm_reorder_static.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-c69cd5e9f8a2202f/rtl/fifo/hwpe_stream_fifo_earlystall.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-c69cd5e9f8a2202f/rtl/fifo/hwpe_stream_fifo_earlystall_sidech.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-c69cd5e9f8a2202f/rtl/fifo/hwpe_stream_fifo_scm_test_wrap.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-c69cd5e9f8a2202f/rtl/fifo/hwpe_stream_fifo_sidech.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-c69cd5e9f8a2202f/rtl/fifo/hwpe_stream_fifo.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-c69cd5e9f8a2202f/rtl/tcdm/hwpe_stream_tcdm_fifo_load_sidech.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-c69cd5e9f8a2202f/rtl/fifo/hwpe_stream_fifo_passthrough.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-c69cd5e9f8a2202f/rtl/streamer/hwpe_stream_source.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-c69cd5e9f8a2202f/rtl/tcdm/hwpe_stream_tcdm_fifo.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-c69cd5e9f8a2202f/rtl/tcdm/hwpe_stream_tcdm_fifo_load.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-c69cd5e9f8a2202f/rtl/tcdm/hwpe_stream_tcdm_fifo_store.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-stream-c69cd5e9f8a2202f/rtl/streamer/hwpe_stream_sink.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"
lappend search_path "$ROOT/.bender/git/checkouts/obi-940bae8c839c275c/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/obi-940bae8c839c275c/src/obi_pkg.sv" \
        "$ROOT/.bender/git/checkouts/obi-940bae8c839c275c/src/obi_intf.sv" \
        "$ROOT/.bender/git/checkouts/obi-940bae8c839c275c/src/obi_rready_converter.sv" \
        "$ROOT/.bender/git/checkouts/obi-940bae8c839c275c/src/apb_to_obi.sv" \
        "$ROOT/.bender/git/checkouts/obi-940bae8c839c275c/src/obi_atop_resolver.sv" \
        "$ROOT/.bender/git/checkouts/obi-940bae8c839c275c/src/obi_cut.sv" \
        "$ROOT/.bender/git/checkouts/obi-940bae8c839c275c/src/obi_demux.sv" \
        "$ROOT/.bender/git/checkouts/obi-940bae8c839c275c/src/obi_err_sbr.sv" \
        "$ROOT/.bender/git/checkouts/obi-940bae8c839c275c/src/obi_mux.sv" \
        "$ROOT/.bender/git/checkouts/obi-940bae8c839c275c/src/obi_sram_shim.sv" \
        "$ROOT/.bender/git/checkouts/obi-940bae8c839c275c/src/obi_xbar.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/apb-967c1a20a8f82c97/include"
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"
lappend search_path "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/include"
lappend search_path "$ROOT/hw/vendor/pulp_platform_axi/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/src/reg_intf.sv" \
        "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/vendor/lowrisc_opentitan/src/prim_subreg_arb.sv" \
        "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/vendor/lowrisc_opentitan/src/prim_subreg_ext.sv" \
        "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/src/apb_to_reg.sv" \
        "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/src/axi_lite_to_reg.sv" \
        "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/src/axi_to_reg_v2.sv" \
        "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/src/periph_to_reg.sv" \
        "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/src/reg_cdc.sv" \
        "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/src/reg_cut.sv" \
        "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/src/reg_demux.sv" \
        "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/src/reg_err_slv.sv" \
        "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/src/reg_filter_empty_writes.sv" \
        "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/src/reg_mux.sv" \
        "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/src/reg_to_apb.sv" \
        "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/src/reg_to_mem.sv" \
        "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/src/reg_to_tlul.sv" \
        "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/src/reg_to_axi.sv" \
        "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/src/reg_uniform.sv" \
        "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/vendor/lowrisc_opentitan/src/prim_subreg_shadow.sv" \
        "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/vendor/lowrisc_opentitan/src/prim_subreg.sv" \
        "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/src/deprecated/axi_to_reg.sv" \
    ]
]} {return 1}

set search_path $search_path_initial

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/scm-d89f244665221304/latch_scm/register_file_1r_1w_test_wrap.sv" \
        "$ROOT/.bender/git/checkouts/scm-d89f244665221304/latch_scm/register_file_1w_64b_multi_port_read_32b_1row.sv" \
        "$ROOT/.bender/git/checkouts/scm-d89f244665221304/latch_scm/register_file_1w_multi_port_read_1row.sv" \
        "$ROOT/.bender/git/checkouts/scm-d89f244665221304/latch_scm/register_file_1r_1w_all.sv" \
        "$ROOT/.bender/git/checkouts/scm-d89f244665221304/latch_scm/register_file_1r_1w_all_test_wrap.sv" \
        "$ROOT/.bender/git/checkouts/scm-d89f244665221304/latch_scm/register_file_1r_1w_be.sv" \
        "$ROOT/.bender/git/checkouts/scm-d89f244665221304/latch_scm/register_file_1r_1w.sv" \
        "$ROOT/.bender/git/checkouts/scm-d89f244665221304/latch_scm/register_file_1r_1w_1row.sv" \
        "$ROOT/.bender/git/checkouts/scm-d89f244665221304/latch_scm/register_file_1w_128b_multi_port_read_32b.sv" \
        "$ROOT/.bender/git/checkouts/scm-d89f244665221304/latch_scm/register_file_1w_64b_multi_port_read_32b.sv" \
        "$ROOT/.bender/git/checkouts/scm-d89f244665221304/latch_scm/register_file_1w_64b_1r_32b.sv" \
        "$ROOT/.bender/git/checkouts/scm-d89f244665221304/latch_scm/register_file_1w_multi_port_read_be.sv" \
        "$ROOT/.bender/git/checkouts/scm-d89f244665221304/latch_scm/register_file_1w_multi_port_read.sv" \
        "$ROOT/.bender/git/checkouts/scm-d89f244665221304/latch_scm/register_file_2r_1w_asymm.sv" \
        "$ROOT/.bender/git/checkouts/scm-d89f244665221304/latch_scm/register_file_2r_1w_asymm_test_wrap.sv" \
        "$ROOT/.bender/git/checkouts/scm-d89f244665221304/latch_scm/register_file_2r_2w.sv" \
        "$ROOT/.bender/git/checkouts/scm-d89f244665221304/latch_scm/register_file_3r_2w.sv" \
        "$ROOT/.bender/git/checkouts/scm-d89f244665221304/latch_scm/register_file_3r_2w_be.sv" \
        "$ROOT/.bender/git/checkouts/scm-d89f244665221304/latch_scm/register_file_multi_way_1w_64b_multi_port_read_32b.sv" \
        "$ROOT/.bender/git/checkouts/scm-d89f244665221304/latch_scm/register_file_multi_way_1w_multi_port_read.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"
lappend search_path "$ROOT/hw/vendor/pulp_platform_axi/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/axi_riscv_atomics-9860e6c21c013eaa/src/axi_res_tbl.sv" \
        "$ROOT/.bender/git/checkouts/axi_riscv_atomics-9860e6c21c013eaa/src/axi_riscv_amos_alu.sv" \
        "$ROOT/.bender/git/checkouts/axi_riscv_atomics-9860e6c21c013eaa/src/axi_riscv_amos.sv" \
        "$ROOT/.bender/git/checkouts/axi_riscv_atomics-9860e6c21c013eaa/src/axi_riscv_lrsc.sv" \
        "$ROOT/.bender/git/checkouts/axi_riscv_atomics-9860e6c21c013eaa/src/axi_riscv_atomics.sv" \
        "$ROOT/.bender/git/checkouts/axi_riscv_atomics-9860e6c21c013eaa/src/axi_riscv_lrsc_wrap.sv" \
        "$ROOT/.bender/git/checkouts/axi_riscv_atomics-9860e6c21c013eaa/src/axi_riscv_amos_wrap.sv" \
        "$ROOT/.bender/git/checkouts/axi_riscv_atomics-9860e6c21c013eaa/src/axi_riscv_atomics_wrap.sv" \
        "$ROOT/.bender/git/checkouts/axi_riscv_atomics-9860e6c21c013eaa/src/axi_riscv_atomics_structs.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"
lappend search_path "$ROOT/hw/vendor/pulp_platform_axi/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/cluster_icache-f368b913e08c6cec/src/snitch_icache_pkg.sv" \
        "$ROOT/.bender/git/checkouts/cluster_icache-f368b913e08c6cec/src/riscv_instr_branch.sv" \
        "$ROOT/.bender/git/checkouts/cluster_icache-f368b913e08c6cec/src/multi_accept_rr_arb.sv" \
        "$ROOT/.bender/git/checkouts/cluster_icache-f368b913e08c6cec/src/snitch_axi_to_cache.sv" \
        "$ROOT/.bender/git/checkouts/cluster_icache-f368b913e08c6cec/src/snitch_icache_l0.sv" \
        "$ROOT/.bender/git/checkouts/cluster_icache-f368b913e08c6cec/src/snitch_icache_refill.sv" \
        "$ROOT/.bender/git/checkouts/cluster_icache-f368b913e08c6cec/src/snitch_icache_lfsr.sv" \
        "$ROOT/.bender/git/checkouts/cluster_icache-f368b913e08c6cec/src/snitch_icache_tag.sv" \
        "$ROOT/.bender/git/checkouts/cluster_icache-f368b913e08c6cec/src/snitch_icache_data.sv" \
        "$ROOT/.bender/git/checkouts/cluster_icache-f368b913e08c6cec/src/snitch_icache_lookup_parallel.sv" \
        "$ROOT/.bender/git/checkouts/cluster_icache-f368b913e08c6cec/src/snitch_icache_handler.sv" \
        "$ROOT/.bender/git/checkouts/cluster_icache-f368b913e08c6cec/src/snitch_icache.sv" \
        "$ROOT/.bender/git/checkouts/cluster_icache-f368b913e08c6cec/src/snitch_read_only_cache.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/fpnew-12c999969330449e/src/fpnew_pkg.sv" \
        "$ROOT/.bender/git/checkouts/fpnew-12c999969330449e/src/fpnew_cast_multi.sv" \
        "$ROOT/.bender/git/checkouts/fpnew-12c999969330449e/src/fpnew_classifier.sv" \
        "$ROOT/.bender/git/checkouts/fpnew-12c999969330449e/vendor/opene906/E906_RTL_FACTORY/gen_rtl/clk/rtl/gated_clk_cell.v" \
        "$ROOT/.bender/git/checkouts/fpnew-12c999969330449e/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_ctrl.v" \
        "$ROOT/.bender/git/checkouts/fpnew-12c999969330449e/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_ff1.v" \
        "$ROOT/.bender/git/checkouts/fpnew-12c999969330449e/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_pack_single.v" \
        "$ROOT/.bender/git/checkouts/fpnew-12c999969330449e/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_prepare.v" \
        "$ROOT/.bender/git/checkouts/fpnew-12c999969330449e/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_round_single.v" \
        "$ROOT/.bender/git/checkouts/fpnew-12c999969330449e/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_special.v" \
        "$ROOT/.bender/git/checkouts/fpnew-12c999969330449e/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_srt_single.v" \
        "$ROOT/.bender/git/checkouts/fpnew-12c999969330449e/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fdsu/rtl/pa_fdsu_top.v" \
        "$ROOT/.bender/git/checkouts/fpnew-12c999969330449e/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fpu/rtl/pa_fpu_dp.v" \
        "$ROOT/.bender/git/checkouts/fpnew-12c999969330449e/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fpu/rtl/pa_fpu_frbus.v" \
        "$ROOT/.bender/git/checkouts/fpnew-12c999969330449e/vendor/opene906/E906_RTL_FACTORY/gen_rtl/fpu/rtl/pa_fpu_src_type.v" \
        "$ROOT/.bender/git/checkouts/fpnew-12c999969330449e/src/fpnew_divsqrt_th_32.sv" \
        "$ROOT/.bender/git/checkouts/fpnew-12c999969330449e/src/fpnew_divsqrt_multi.sv" \
        "$ROOT/.bender/git/checkouts/fpnew-12c999969330449e/src/fpnew_fma.sv" \
        "$ROOT/.bender/git/checkouts/fpnew-12c999969330449e/src/fpnew_fma_multi.sv" \
        "$ROOT/.bender/git/checkouts/fpnew-12c999969330449e/src/fpnew_sdotp_multi.sv" \
        "$ROOT/.bender/git/checkouts/fpnew-12c999969330449e/src/fpnew_sdotp_multi_wrapper.sv" \
        "$ROOT/.bender/git/checkouts/fpnew-12c999969330449e/src/fpnew_noncomp.sv" \
        "$ROOT/.bender/git/checkouts/fpnew-12c999969330449e/src/fpnew_opgroup_block.sv" \
        "$ROOT/.bender/git/checkouts/fpnew-12c999969330449e/src/fpnew_opgroup_fmt_slice.sv" \
        "$ROOT/.bender/git/checkouts/fpnew-12c999969330449e/src/fpnew_opgroup_multifmt_slice.sv" \
        "$ROOT/.bender/git/checkouts/fpnew-12c999969330449e/src/fpnew_rounding.sv" \
        "$ROOT/.bender/git/checkouts/fpnew-12c999969330449e/src/lfsr_sr.sv" \
        "$ROOT/.bender/git/checkouts/fpnew-12c999969330449e/src/fpnew_top.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/hwpe-mac-engine-828f13659ef3c707/rtl"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/hwpe-mac-engine-828f13659ef3c707/rtl/mac_package.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-mac-engine-828f13659ef3c707/rtl/mac_engine.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-mac-engine-828f13659ef3c707/rtl/mac_fsm.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-mac-engine-828f13659ef3c707/rtl/mac_streamer.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-mac-engine-828f13659ef3c707/rtl/mac_ctrl.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-mac-engine-828f13659ef3c707/rtl/mac_top.sv" \
        "$ROOT/.bender/git/checkouts/hwpe-mac-engine-828f13659ef3c707/wrap/mac_top_wrap.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/axi_stream-683bf88f199f0ed8/include"
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"
lappend search_path "$ROOT/.bender/git/checkouts/idma-77ce09db65eb3f4d/src/include"
lappend search_path "$ROOT/.bender/git/checkouts/idma-77ce09db65eb3f4d/target/rtl/include"
lappend search_path "$ROOT/.bender/git/checkouts/idma-77ce09db65eb3f4d/test"
lappend search_path "$ROOT/.bender/git/checkouts/obi-940bae8c839c275c/include"
lappend search_path "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/include"
lappend search_path "$ROOT/hw/vendor/pulp_platform_axi/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/idma-77ce09db65eb3f4d/src/idma_pkg.sv" \
        "$ROOT/.bender/git/checkouts/idma-77ce09db65eb3f4d/src/backend/idma_axil_read.sv" \
        "$ROOT/.bender/git/checkouts/idma-77ce09db65eb3f4d/src/backend/idma_axil_write.sv" \
        "$ROOT/.bender/git/checkouts/idma-77ce09db65eb3f4d/src/backend/idma_axi_read.sv" \
        "$ROOT/.bender/git/checkouts/idma-77ce09db65eb3f4d/src/backend/idma_axi_write.sv" \
        "$ROOT/.bender/git/checkouts/idma-77ce09db65eb3f4d/src/backend/idma_axis_read.sv" \
        "$ROOT/.bender/git/checkouts/idma-77ce09db65eb3f4d/src/backend/idma_axis_write.sv" \
        "$ROOT/.bender/git/checkouts/idma-77ce09db65eb3f4d/src/backend/idma_channel_coupler.sv" \
        "$ROOT/.bender/git/checkouts/idma-77ce09db65eb3f4d/src/backend/idma_dataflow_element.sv" \
        "$ROOT/.bender/git/checkouts/idma-77ce09db65eb3f4d/src/backend/idma_error_handler.sv" \
        "$ROOT/.bender/git/checkouts/idma-77ce09db65eb3f4d/src/backend/idma_init_read.sv" \
        "$ROOT/.bender/git/checkouts/idma-77ce09db65eb3f4d/src/backend/idma_init_write.sv" \
        "$ROOT/.bender/git/checkouts/idma-77ce09db65eb3f4d/src/backend/idma_legalizer_page_splitter.sv" \
        "$ROOT/.bender/git/checkouts/idma-77ce09db65eb3f4d/src/backend/idma_legalizer_pow2_splitter.sv" \
        "$ROOT/.bender/git/checkouts/idma-77ce09db65eb3f4d/src/backend/idma_obi_read.sv" \
        "$ROOT/.bender/git/checkouts/idma-77ce09db65eb3f4d/src/backend/idma_obi_write.sv" \
        "$ROOT/.bender/git/checkouts/idma-77ce09db65eb3f4d/src/backend/idma_tilelink_read.sv" \
        "$ROOT/.bender/git/checkouts/idma-77ce09db65eb3f4d/src/backend/idma_tilelink_write.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/axi_stream-683bf88f199f0ed8/include"
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"
lappend search_path "$ROOT/.bender/git/checkouts/idma-77ce09db65eb3f4d/src/include"
lappend search_path "$ROOT/.bender/git/checkouts/idma-77ce09db65eb3f4d/target/rtl/include"
lappend search_path "$ROOT/.bender/git/checkouts/idma-77ce09db65eb3f4d/test"
lappend search_path "$ROOT/.bender/git/checkouts/obi-940bae8c839c275c/include"
lappend search_path "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/include"
lappend search_path "$ROOT/hw/vendor/pulp_platform_axi/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/idma-77ce09db65eb3f4d/target/rtl/idma_generated.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/axi_stream-683bf88f199f0ed8/include"
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"
lappend search_path "$ROOT/.bender/git/checkouts/idma-77ce09db65eb3f4d/src/include"
lappend search_path "$ROOT/.bender/git/checkouts/idma-77ce09db65eb3f4d/target/rtl/include"
lappend search_path "$ROOT/.bender/git/checkouts/idma-77ce09db65eb3f4d/test"
lappend search_path "$ROOT/.bender/git/checkouts/obi-940bae8c839c275c/include"
lappend search_path "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/include"
lappend search_path "$ROOT/hw/vendor/pulp_platform_axi/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/idma-77ce09db65eb3f4d/src/midend/idma_mp_dist_midend.sv" \
        "$ROOT/.bender/git/checkouts/idma-77ce09db65eb3f4d/src/midend/idma_mp_split_midend.sv" \
        "$ROOT/.bender/git/checkouts/idma-77ce09db65eb3f4d/src/midend/idma_nd_midend.sv" \
        "$ROOT/.bender/git/checkouts/idma-77ce09db65eb3f4d/src/midend/idma_rt_midend.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/axi_stream-683bf88f199f0ed8/include"
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"
lappend search_path "$ROOT/.bender/git/checkouts/idma-77ce09db65eb3f4d/src/include"
lappend search_path "$ROOT/.bender/git/checkouts/idma-77ce09db65eb3f4d/target/rtl/include"
lappend search_path "$ROOT/.bender/git/checkouts/idma-77ce09db65eb3f4d/test"
lappend search_path "$ROOT/.bender/git/checkouts/obi-940bae8c839c275c/include"
lappend search_path "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/include"
lappend search_path "$ROOT/hw/vendor/pulp_platform_axi/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/idma-77ce09db65eb3f4d/src/frontend/desc64/idma_desc64_ar_gen.sv" \
        "$ROOT/.bender/git/checkouts/idma-77ce09db65eb3f4d/src/frontend/desc64/idma_desc64_ar_gen_prefetch.sv" \
        "$ROOT/.bender/git/checkouts/idma-77ce09db65eb3f4d/src/frontend/desc64/idma_desc64_reader.sv" \
        "$ROOT/.bender/git/checkouts/idma-77ce09db65eb3f4d/src/frontend/desc64/idma_desc64_reader_gater.sv" \
        "$ROOT/.bender/git/checkouts/idma-77ce09db65eb3f4d/src/frontend/desc64/idma_desc64_reshaper.sv" \
        "$ROOT/.bender/git/checkouts/idma-77ce09db65eb3f4d/src/frontend/idma_transfer_id_gen.sv" \
        "$ROOT/.bender/git/checkouts/idma-77ce09db65eb3f4d/src/frontend/desc64/idma_desc64_reg_wrapper.sv" \
        "$ROOT/.bender/git/checkouts/idma-77ce09db65eb3f4d/src/frontend/desc64/idma_desc64_top.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/riscv-dbg-0edb7716f6e62c05/src/dm_pkg.sv" \
        "$ROOT/.bender/git/checkouts/riscv-dbg-0edb7716f6e62c05/debug_rom/debug_rom.sv" \
        "$ROOT/.bender/git/checkouts/riscv-dbg-0edb7716f6e62c05/debug_rom/debug_rom_one_scratch.sv" \
        "$ROOT/.bender/git/checkouts/riscv-dbg-0edb7716f6e62c05/src/dm_csrs.sv" \
        "$ROOT/.bender/git/checkouts/riscv-dbg-0edb7716f6e62c05/src/dm_mem.sv" \
        "$ROOT/.bender/git/checkouts/riscv-dbg-0edb7716f6e62c05/src/dmi_cdc.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/riscv-dbg-0edb7716f6e62c05/src/dmi_jtag_tap.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/riscv-dbg-0edb7716f6e62c05/src/dm_sba.sv" \
        "$ROOT/.bender/git/checkouts/riscv-dbg-0edb7716f6e62c05/src/dm_top.sv" \
        "$ROOT/.bender/git/checkouts/riscv-dbg-0edb7716f6e62c05/src/dmi_jtag.sv" \
        "$ROOT/.bender/git/checkouts/riscv-dbg-0edb7716f6e62c05/src/dm_obi_top.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"
lappend search_path "$ROOT/hw/vendor/pulp_platform_axi/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/xdma_axi_adapter-c2b30a75f7002c7e/src/xdma_pkg.sv" \
        "$ROOT/.bender/git/checkouts/xdma_axi_adapter-c2b30a75f7002c7e/src/find_first_one_idx.sv" \
        "$ROOT/.bender/git/checkouts/xdma_axi_adapter-c2b30a75f7002c7e/src/xdma_req_manager.sv" \
        "$ROOT/.bender/git/checkouts/xdma_axi_adapter-c2b30a75f7002c7e/src/xdma_grant_manager.sv" \
        "$ROOT/.bender/git/checkouts/xdma_axi_adapter-c2b30a75f7002c7e/src/xdma_data_path.sv" \
        "$ROOT/.bender/git/checkouts/xdma_axi_adapter-c2b30a75f7002c7e/src/xdma_burst_reshaper.sv" \
        "$ROOT/.bender/git/checkouts/xdma_axi_adapter-c2b30a75f7002c7e/src/xdma_meta_manager.sv" \
        "$ROOT/.bender/git/checkouts/xdma_axi_adapter-c2b30a75f7002c7e/src/xdma_axi_to_write.sv" \
        "$ROOT/.bender/git/checkouts/xdma_axi_adapter-c2b30a75f7002c7e/src/xdma_write_demux.sv" \
        "$ROOT/.bender/git/checkouts/xdma_axi_adapter-c2b30a75f7002c7e/src/xdma_finish_manager.sv" \
        "$ROOT/.bender/git/checkouts/xdma_axi_adapter-c2b30a75f7002c7e/src/xdma_req_backend.sv" \
        "$ROOT/.bender/git/checkouts/xdma_axi_adapter-c2b30a75f7002c7e/src/xdma_axi_adapter_top.sv" \
    ]
]} {return 1}

set search_path $search_path_initial

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/apb_timer-7f0a4fb67c0644d7/src/timer.sv" \
        "$ROOT/.bender/git/checkouts/apb_timer-7f0a4fb67c0644d7/src/apb_timer.sv" \
    ]
]} {return 1}

set search_path $search_path_initial

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/apb_uart-3faf55e082d60fde/src/slib_clock_div.sv" \
        "$ROOT/.bender/git/checkouts/apb_uart-3faf55e082d60fde/src/slib_counter.sv" \
        "$ROOT/.bender/git/checkouts/apb_uart-3faf55e082d60fde/src/slib_edge_detect.sv" \
        "$ROOT/.bender/git/checkouts/apb_uart-3faf55e082d60fde/src/slib_fifo.sv" \
        "$ROOT/.bender/git/checkouts/apb_uart-3faf55e082d60fde/src/slib_input_filter.sv" \
        "$ROOT/.bender/git/checkouts/apb_uart-3faf55e082d60fde/src/slib_input_sync.sv" \
        "$ROOT/.bender/git/checkouts/apb_uart-3faf55e082d60fde/src/slib_mv_filter.sv" \
        "$ROOT/.bender/git/checkouts/apb_uart-3faf55e082d60fde/src/uart_baudgen.sv" \
        "$ROOT/.bender/git/checkouts/apb_uart-3faf55e082d60fde/src/uart_interrupt.sv" \
        "$ROOT/.bender/git/checkouts/apb_uart-3faf55e082d60fde/src/uart_receiver.sv" \
        "$ROOT/.bender/git/checkouts/apb_uart-3faf55e082d60fde/src/uart_transmitter.sv" \
        "$ROOT/.bender/git/checkouts/apb_uart-3faf55e082d60fde/src/apb_uart.sv" \
        "$ROOT/.bender/git/checkouts/apb_uart-3faf55e082d60fde/src/apb_uart_wrap.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"
lappend search_path "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/include"
lappend search_path "$ROOT/hw/vendor/pulp_platform_axi/include"
lappend search_path "$ROOT/hw/vendor/pulp_platform_axi_tlb/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/hw/vendor/pulp_platform_axi_tlb/src/axi_tlb_l1_chan.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi_tlb/src/axi_tlb_reg_pkg.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi_tlb/src/axi_tlb_reg_top.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi_tlb/src/axi_tlb_l1.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi_tlb/src/axi_tlb_noreg.sv" \
        "$ROOT/hw/vendor/pulp_platform_axi_tlb/src/axi_tlb.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"
lappend search_path "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/clint-b5738b1472f66607/src/clint_reg_pkg.sv" \
        "$ROOT/.bender/git/checkouts/clint-b5738b1472f66607/src/clint_reg_top.sv" \
        "$ROOT/.bender/git/checkouts/clint-b5738b1472f66607/src/clint.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"
lappend search_path "$ROOT/hw/vendor/pulp_platform_axi/include"

if {0 == [analyze -format sv \
    -define { \
        WB_DCACHE=1 \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/include/cv64a6_imafdc_sv39_config_pkg.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/include/riscv_pkg.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/include/ariane_dm_pkg.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/include/ariane_pkg.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/mmu_sv39/tlb.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/mmu_sv39/mmu.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/mmu_sv39/ptw.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"
lappend search_path "$ROOT/hw/vendor/pulp_platform_axi/include"

if {0 == [analyze -format sv \
    -define { \
        WB_DCACHE=1 \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/include/ariane_rvfi_pkg.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/include/ariane_axi_pkg.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/include/wt_cache_pkg.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/include/std_cache_pkg.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/include/axi_intf.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/include/cvxif_pkg.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/cvxif_example/include/cvxif_instr_pkg.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/cvxif_fu.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/cvxif_example/cvxif_example_coprocessor.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/cvxif_example/instr_decoder.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/ariane.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/cva6.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/alu.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/fpu_wrap.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/branch_unit.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/compressed_decoder.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/controller.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/csr_buffer.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/csr_regfile.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/decoder.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/ex_stage.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/instr_realign.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/id_stage.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/issue_read_operands.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/issue_stage.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/load_unit.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/load_store_unit.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/lsu_bypass.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/mult.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/multiplier.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/serdiv.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/perf_counters.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/ariane_regfile_ff.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/ariane_regfile_fpga.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/re_name.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/scoreboard.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/store_buffer.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/amo_buffer.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/store_unit.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/commit_stage.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/axi_shim.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/frontend/btb.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/frontend/bht.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/frontend/ras.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/frontend/instr_scan.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/frontend/instr_queue.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/frontend/frontend.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/cache_subsystem/wt_dcache_ctrl.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/cache_subsystem/wt_dcache_mem.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/cache_subsystem/wt_dcache_missunit.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/cache_subsystem/wt_dcache_wbuffer.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/cache_subsystem/wt_dcache.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/cache_subsystem/cva6_icache_tag.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/cache_subsystem/cva6_icache_data.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/cache_subsystem/cva6_icache.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/cache_subsystem/wt_cache_subsystem.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/cache_subsystem/wt_axi_adapter.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/cache_subsystem/tag_cmp.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/cache_subsystem/cache_ctrl.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/cache_subsystem/amo_alu.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/cache_subsystem/wt_l15_adapter.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/corev_apu/tb/axi_adapter.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/cache_subsystem/miss_handler.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/cache_subsystem/std_nbdcache_data.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/cache_subsystem/std_nbdcache_tag.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/cache_subsystem/std_nbdcache_valid_dirty.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/cache_subsystem/std_nbdcache.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/cache_subsystem/cva6_icache_axi_wrapper.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/cache_subsystem/std_cache_subsystem.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/pmp/src/pmp.sv" \
        "$ROOT/hw/vendor/openhwgroup_cva6/core/pmp/src/pmp_entry.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"
lappend search_path "$ROOT/hw/vendor/openhwgroup_cva6/common/local/util"
lappend search_path "$ROOT/hw/vendor/pulp_platform_axi/include"

if {0 == [analyze -format sv \
    -define { \
        WB_DCACHE=1 \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/hw/vendor/openhwgroup_cva6/common/local/util/sram.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"
lappend search_path "$ROOT/hw/vendor/openhwgroup_cva6/common/local/util"
lappend search_path "$ROOT/hw/vendor/pulp_platform_axi/include"

if {0 == [analyze -format sv \
    -define { \
        WB_DCACHE=1 \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/hw/vendor/openhwgroup_cva6/common/local/util/tc_sram_wrapper.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"
lappend search_path "$ROOT/.bender/git/checkouts/floo_noc-696cf2b1eb86a8f9/hw/include"
lappend search_path "$ROOT/.bender/git/checkouts/idma-77ce09db65eb3f4d/src/include"
lappend search_path "$ROOT/.bender/git/checkouts/idma-77ce09db65eb3f4d/target/rtl/include"
lappend search_path "$ROOT/.bender/git/checkouts/idma-77ce09db65eb3f4d/test"
lappend search_path "$ROOT/hw/vendor/pulp_platform_axi/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/floo_noc-696cf2b1eb86a8f9/hw/floo_pkg.sv" \
        "$ROOT/.bender/git/checkouts/floo_noc-696cf2b1eb86a8f9/hw/floo_cut.sv" \
        "$ROOT/.bender/git/checkouts/floo_noc-696cf2b1eb86a8f9/hw/floo_fifo.sv" \
        "$ROOT/.bender/git/checkouts/floo_noc-696cf2b1eb86a8f9/hw/floo_cdc.sv" \
        "$ROOT/.bender/git/checkouts/floo_noc-696cf2b1eb86a8f9/hw/floo_route_select.sv" \
        "$ROOT/.bender/git/checkouts/floo_noc-696cf2b1eb86a8f9/hw/floo_route_comp.sv" \
        "$ROOT/.bender/git/checkouts/floo_noc-696cf2b1eb86a8f9/hw/floo_vc_arbiter.sv" \
        "$ROOT/.bender/git/checkouts/floo_noc-696cf2b1eb86a8f9/hw/floo_wormhole_arbiter.sv" \
        "$ROOT/.bender/git/checkouts/floo_noc-696cf2b1eb86a8f9/hw/floo_simple_rob.sv" \
        "$ROOT/.bender/git/checkouts/floo_noc-696cf2b1eb86a8f9/hw/floo_rob.sv" \
        "$ROOT/.bender/git/checkouts/floo_noc-696cf2b1eb86a8f9/hw/floo_rob_wrapper.sv" \
        "$ROOT/.bender/git/checkouts/floo_noc-696cf2b1eb86a8f9/hw/floo_meta_buffer.sv" \
        "$ROOT/.bender/git/checkouts/floo_noc-696cf2b1eb86a8f9/hw/floo_nw_join.sv" \
        "$ROOT/.bender/git/checkouts/floo_noc-696cf2b1eb86a8f9/hw/floo_axi_chimney.sv" \
        "$ROOT/.bender/git/checkouts/floo_noc-696cf2b1eb86a8f9/hw/floo_nw_chimney.sv" \
        "$ROOT/.bender/git/checkouts/floo_noc-696cf2b1eb86a8f9/hw/floo_router.sv" \
        "$ROOT/.bender/git/checkouts/floo_noc-696cf2b1eb86a8f9/hw/floo_axi_router.sv" \
        "$ROOT/.bender/git/checkouts/floo_noc-696cf2b1eb86a8f9/hw/floo_nw_router.sv" \
        "$ROOT/.bender/git/checkouts/floo_noc-696cf2b1eb86a8f9/hw/vc_router_util/floo_credit_counter.sv" \
        "$ROOT/.bender/git/checkouts/floo_noc-696cf2b1eb86a8f9/hw/vc_router_util/floo_input_fifo.sv" \
        "$ROOT/.bender/git/checkouts/floo_noc-696cf2b1eb86a8f9/hw/vc_router_util/floo_input_port.sv" \
        "$ROOT/.bender/git/checkouts/floo_noc-696cf2b1eb86a8f9/hw/vc_router_util/floo_look_ahead_routing.sv" \
        "$ROOT/.bender/git/checkouts/floo_noc-696cf2b1eb86a8f9/hw/vc_router_util/floo_mux.sv" \
        "$ROOT/.bender/git/checkouts/floo_noc-696cf2b1eb86a8f9/hw/vc_router_util/floo_rr_arbiter.sv" \
        "$ROOT/.bender/git/checkouts/floo_noc-696cf2b1eb86a8f9/hw/vc_router_util/floo_sa_global.sv" \
        "$ROOT/.bender/git/checkouts/floo_noc-696cf2b1eb86a8f9/hw/vc_router_util/floo_sa_local.sv" \
        "$ROOT/.bender/git/checkouts/floo_noc-696cf2b1eb86a8f9/hw/vc_router_util/floo_vc_assignment.sv" \
        "$ROOT/.bender/git/checkouts/floo_noc-696cf2b1eb86a8f9/hw/vc_router_util/floo_vc_router_switch.sv" \
        "$ROOT/.bender/git/checkouts/floo_noc-696cf2b1eb86a8f9/hw/vc_router_util/floo_vc_selection.sv" \
        "$ROOT/.bender/git/checkouts/floo_noc-696cf2b1eb86a8f9/hw/floo_vc_router.sv" \
        "$ROOT/.bender/git/checkouts/floo_noc-696cf2b1eb86a8f9/hw/floo_nw_vc_chimney.sv" \
        "$ROOT/.bender/git/checkouts/floo_noc-696cf2b1eb86a8f9/hw/floo_nw_vc_router.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"
lappend search_path "$ROOT/hw/vendor/pulp_platform_axi/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/hemaia_axi_spi_slave-a6decc0d5c4c4904/occamy_spi_slave.sv" \
        "$ROOT/.bender/git/checkouts/hemaia_axi_spi_slave-a6decc0d5c4c4904/axi_spi_slave.sv" \
        "$ROOT/.bender/git/checkouts/hemaia_axi_spi_slave-a6decc0d5c4c4904/spi_slave_axi_plug.sv" \
        "$ROOT/.bender/git/checkouts/hemaia_axi_spi_slave-a6decc0d5c4c4904/spi_slave_cmd_parser.sv" \
        "$ROOT/.bender/git/checkouts/hemaia_axi_spi_slave-a6decc0d5c4c4904/spi_slave_controller.sv" \
        "$ROOT/.bender/git/checkouts/hemaia_axi_spi_slave-a6decc0d5c4c4904/spi_slave_dc_fifo.sv" \
        "$ROOT/.bender/git/checkouts/hemaia_axi_spi_slave-a6decc0d5c4c4904/spi_slave_regs.sv" \
        "$ROOT/.bender/git/checkouts/hemaia_axi_spi_slave-a6decc0d5c4c4904/spi_slave_rx.sv" \
        "$ROOT/.bender/git/checkouts/hemaia_axi_spi_slave-a6decc0d5c4c4904/spi_slave_syncro.sv" \
        "$ROOT/.bender/git/checkouts/hemaia_axi_spi_slave-a6decc0d5c4c4904/spi_slave_tx.sv" \
        "$ROOT/.bender/git/checkouts/hemaia_axi_spi_slave-a6decc0d5c4c4904/axi_slice_dc/dc_synchronizer.v" \
        "$ROOT/.bender/git/checkouts/hemaia_axi_spi_slave-a6decc0d5c4c4904/axi_slice_dc/dc_token_ring_fifo_din.v" \
        "$ROOT/.bender/git/checkouts/hemaia_axi_spi_slave-a6decc0d5c4c4904/axi_slice_dc/dc_token_ring_fifo_dout.v" \
        "$ROOT/.bender/git/checkouts/hemaia_axi_spi_slave-a6decc0d5c4c4904/axi_slice_dc/dc_token_ring.v" \
        "$ROOT/.bender/git/checkouts/hemaia_axi_spi_slave-a6decc0d5c4c4904/axi_slice_dc/dc_full_detector.v" \
        "$ROOT/.bender/git/checkouts/hemaia_axi_spi_slave-a6decc0d5c4c4904/axi_slice_dc/dc_data_buffer.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"
lappend search_path "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/include"
lappend search_path "$ROOT/hw/vendor/pulp_platform_axi/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/hw/hemaia/hemaia_d2d_link/src/ctrl/hemaia_d2d_link_reg_pkg.sv" \
        "$ROOT/hw/hemaia/hemaia_d2d_link/src/ctrl/hemaia_d2d_link_reg_top.sv" \
        "$ROOT/hw/hemaia/hemaia_d2d_link/src/ctrl/hemaia_d2d_link_reset_controller.sv" \
        "$ROOT/hw/hemaia/hemaia_d2d_link/src/hemaia_d2d_link_pkg.sv" \
        "$ROOT/hw/hemaia/hemaia_d2d_link/src/common/hemaia_d2d_link_stream_fifo.sv" \
        "$ROOT/hw/hemaia/hemaia_d2d_link/src/common/one_bit_prng.sv" \
        "$ROOT/hw/hemaia/hemaia_d2d_link/src/common/BasicCeilingCounter.sv" \
        "$ROOT/hw/hemaia/hemaia_d2d_link/src/common/NonOverflowCounter.sv" \
        "$ROOT/hw/hemaia/hemaia_d2d_link/src/common/ComplexQueue_Merge.sv" \
        "$ROOT/hw/hemaia/hemaia_d2d_link/src/common/ComplexQueue_Scatter.sv" \
        "$ROOT/hw/hemaia/hemaia_d2d_link/src/common/sync_reg.sv" \
        "$ROOT/hw/hemaia/hemaia_d2d_link/src/hemaia_d2d_link.sv" \
        "$ROOT/hw/hemaia/hemaia_d2d_link/src/network_framing/axi_r_chan_remapper.sv" \
        "$ROOT/hw/hemaia/hemaia_d2d_link/src/network_framing/axi_transaction_counter.sv" \
        "$ROOT/hw/hemaia/hemaia_d2d_link/src/network_framing/hemaia_d2d_link_framer.sv" \
        "$ROOT/hw/hemaia/hemaia_d2d_link/src/network_routing/hemaia_d2d_link_stream_splitter.sv" \
        "$ROOT/hw/hemaia/hemaia_d2d_link/src/network_routing/hemaia_d2d_link_stream_mux.sv" \
        "$ROOT/hw/hemaia/hemaia_d2d_link/src/network_routing/hemaia_d2d_link_aw_gate_recorder.sv" \
        "$ROOT/hw/hemaia/hemaia_d2d_link/src/network_routing/hemaia_d2d_link_stream_gatekeeper.sv" \
        "$ROOT/hw/hemaia/hemaia_d2d_link/src/network_routing/hemaia_d2d_link_router_switching_fabric.sv" \
        "$ROOT/hw/hemaia/hemaia_d2d_link/src/network_routing/hemaia_d2d_link_router_rc.sv" \
        "$ROOT/hw/hemaia/hemaia_d2d_link/src/network_routing/hemaia_d2d_link_router.sv" \
        "$ROOT/hw/hemaia/hemaia_d2d_link/src/data_link/hemaia_d2d_link_fec_encoder.sv" \
        "$ROOT/hw/hemaia/hemaia_d2d_link/src/data_link/hemaia_d2d_link_fec_decoder.sv" \
        "$ROOT/hw/hemaia/hemaia_d2d_link/src/data_link/hemaia_d2d_link_data_link.sv" \
        "$ROOT/hw/hemaia/hemaia_d2d_link/src/phy_digital/hemaia_d2d_link_lfsr.sv" \
        "$ROOT/hw/hemaia/hemaia_d2d_link/src/phy_digital/hemaia_d2d_link_coherence_checker.sv" \
        "$ROOT/hw/hemaia/hemaia_d2d_link/src/phy_digital/hemaia_d2d_link_phy_prbs_generator.sv" \
        "$ROOT/hw/hemaia/hemaia_d2d_link/src/phy_digital/hemaia_d2d_link_phy_prbs_checker.sv" \
        "$ROOT/hw/hemaia/hemaia_d2d_link/src/phy_digital/hemaia_d2d_link_clk_mux_stage1.sv" \
        "$ROOT/hw/hemaia/hemaia_d2d_link/src/phy_digital/hemaia_d2d_link_clk_mux_stage2.sv" \
        "$ROOT/hw/hemaia/hemaia_d2d_link/src/phy_digital/hemaia_clock_delay_controller.sv" \
        "$ROOT/hw/hemaia/hemaia_d2d_link/src/phy_digital/hemaia_d2d_link_phy_rx_fsm_des.sv" \
        "$ROOT/hw/hemaia/hemaia_d2d_link/src/phy_digital/hemaia_d2d_link_phy_rx_link_bypass.sv" \
        "$ROOT/hw/hemaia/hemaia_d2d_link/src/phy_digital/hemaia_d2d_link_phy_tx_fsm.sv" \
        "$ROOT/hw/hemaia/hemaia_d2d_link/src/phy_digital/hemaia_d2d_link_phy_tx_link_bypass.sv" \
        "$ROOT/hw/hemaia/hemaia_d2d_link/src/phy_digital/hemaia_d2d_link_rx_mapper.sv" \
        "$ROOT/hw/hemaia/hemaia_d2d_link/src/phy_digital/hemaia_d2d_link_tx_mapper.sv" \
        "$ROOT/hw/hemaia/hemaia_d2d_link/src/phy_digital/hemaia_d2d_link_phy_digital.sv" \
        "$ROOT/hw/hemaia/hemaia_d2d_link/src/phy_analog/tx_cell.sv" \
        "$ROOT/hw/hemaia/hemaia_d2d_link/src/phy_analog/rx_clk_cell.sv" \
        "$ROOT/hw/hemaia/hemaia_d2d_link/src/phy_analog/rx_cell.sv" \
        "$ROOT/hw/hemaia/hemaia_d2d_link/src/phy_analog/data_io_cell.sv" \
        "$ROOT/hw/hemaia/hemaia_d2d_link/src/phy_analog/clk_recv_io_cell.sv" \
        "$ROOT/hw/hemaia/hemaia_d2d_link/src/phy_analog/hemaia_d2d_link_phy_analog.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"
lappend search_path "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/prim/prim_pulp_platform/prim_flop_2sync.sv" \
        "$ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/prim/rtl/prim_util_pkg.sv" \
        "$ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/prim/rtl/prim_sync_reqack.sv" \
        "$ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/prim/rtl/prim_sync_reqack_data.sv" \
        "$ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/prim/rtl/prim_pulse_sync.sv" \
        "$ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/prim/rtl/prim_packer_fifo.sv" \
        "$ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/prim/rtl/prim_fifo_sync.sv" \
        "$ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/prim/rtl/prim_filter_ctr.sv" \
        "$ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/prim/rtl/prim_intr_hw.sv" \
        "$ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/prim/rtl/prim_fifo_async.sv" \
        "$ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/gpio/rtl/gpio_reg_pkg.sv" \
        "$ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/i2c/rtl/i2c_reg_pkg.sv" \
        "$ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/spi_host/rtl/spi_host_reg_pkg.sv" \
        "$ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/gpio/rtl/gpio_reg_top.sv" \
        "$ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/i2c/rtl/i2c_reg_top.sv" \
        "$ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/spi_host/rtl/spi_host_reg_top.sv" \
        "$ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/i2c/rtl/i2c_fsm.sv" \
        "$ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/rv_plic/rtl/rv_plic_gateway.sv" \
        "$ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/spi_host/rtl/spi_host_byte_merge.sv" \
        "$ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/spi_host/rtl/spi_host_byte_select.sv" \
        "$ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/spi_host/rtl/spi_host_cmd_pkg.sv" \
        "$ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/spi_host/rtl/spi_host_command_cdc.sv" \
        "$ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/spi_host/rtl/spi_host_fsm.sv" \
        "$ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/spi_host/rtl/spi_host_window.sv" \
        "$ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/spi_host/rtl/spi_host_data_cdc.sv" \
        "$ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/spi_host/rtl/spi_host_shift_register.sv" \
        "$ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/i2c/rtl/i2c_core.sv" \
        "$ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/rv_plic/rtl/rv_plic_target.sv" \
        "$ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/spi_host/rtl/spi_host_core.sv" \
        "$ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/gpio/rtl/gpio.sv" \
        "$ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/i2c/rtl/i2c.sv" \
        "$ROOT/hw/vendor/pulp_platform_opentitan_peripherals/src/spi_host/rtl/spi_host.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"
lappend search_path "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/mem_interface/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/reqrsp_interface/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_ssr/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/tcdm_interface/include"
lappend search_path "$ROOT/hw/vendor/pulp_platform_axi/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/future/src/mem_to_axi_lite.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/future/src/idma_reg64_frontend_reg_pkg.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/future/src/idma_tf_id_gen.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/future/src/dma/axi_dma_data_path.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/future/src/axi_interleaved_xbar.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/future/src/axi_zero_mem.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/future/src/idma_reg64_frontend_reg_top.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/future/src/idma_reg64_frontend.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/future/src/dma/axi_dma_data_mover.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/future/src/dma/axi_dma_burst_reshaper.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/future/src/dma/axi_dma_backend.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"
lappend search_path "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/mem_interface/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/reqrsp_interface/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_ssr/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/tcdm_interface/include"
lappend search_path "$ROOT/hw/vendor/pulp_platform_axi/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/reqrsp_interface/src/reqrsp_pkg.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/reqrsp_interface/src/reqrsp_intf.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/reqrsp_interface/src/axi_to_reqrsp.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/reqrsp_interface/src/reqrsp_cut.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/reqrsp_interface/src/reqrsp_demux.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/reqrsp_interface/src/reqrsp_iso.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/reqrsp_interface/src/reqrsp_mux.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/reqrsp_interface/src/reqrsp_to_axi.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"
lappend search_path "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/mem_interface/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/reqrsp_interface/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_ssr/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/tcdm_interface/include"
lappend search_path "$ROOT/hw/vendor/pulp_platform_axi/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/mem_interface/src/mem_wide_narrow_mux.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/mem_interface/src/mem_interface.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"
lappend search_path "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/mem_interface/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/reqrsp_interface/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_ssr/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/tcdm_interface/include"
lappend search_path "$ROOT/hw/vendor/pulp_platform_axi/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/tcdm_interface/src/tcdm_interface.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/tcdm_interface/src/axi_to_tcdm.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/tcdm_interface/src/reqrsp_to_tcdm.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/tcdm_interface/src/tcdm_mux.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"
lappend search_path "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/mem_interface/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/reqrsp_interface/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_ssr/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/tcdm_interface/include"
lappend search_path "$ROOT/hw/vendor/pulp_platform_axi/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch/src/snitch_pma_pkg.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch/src/riscv_instr.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch/src/csr_snax_def.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch/src/snitch_pkg.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch/src/snitch_regfile_ff.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch/src/snitch_lsu.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch/src/snitch_l0_tlb.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"
lappend search_path "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/mem_interface/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/reqrsp_interface/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_ssr/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/tcdm_interface/include"
lappend search_path "$ROOT/hw/vendor/pulp_platform_axi/include"

if {0 == [analyze -format sv \
    -define { \
        SNITCH_ENABLE_PERF \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch/src/snitch.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"
lappend search_path "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/mem_interface/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/reqrsp_interface/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_ssr/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/tcdm_interface/include"
lappend search_path "$ROOT/hw/vendor/pulp_platform_axi/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_vm/src/snitch_ptw.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"
lappend search_path "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/mem_interface/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/reqrsp_interface/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_ssr/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/tcdm_interface/include"
lappend search_path "$ROOT/hw/vendor/pulp_platform_axi/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_dma/src/axi_dma_pkg.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_dma/src/axi_dma_error_handler.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_dma/src/axi_dma_perf_counters.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_dma/src/axi_dma_twod_ext.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_dma/src/axi_dma_tc_snitch_fe.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"
lappend search_path "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/mem_interface/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/reqrsp_interface/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_ssr/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/tcdm_interface/include"
lappend search_path "$ROOT/hw/vendor/pulp_platform_axi/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_ipu/src/snitch_ipu_pkg.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_ipu/src/snitch_ipu_alu.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_ipu/src/snitch_int_ss.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"
lappend search_path "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/mem_interface/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/reqrsp_interface/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_ssr/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/tcdm_interface/include"
lappend search_path "$ROOT/hw/vendor/pulp_platform_axi/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_ssr/src/snitch_ssr_pkg.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_ssr/src/snitch_ssr_switch.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_ssr/src/snitch_ssr_credit_counter.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_ssr/src/snitch_ssr_indirector.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_ssr/src/snitch_ssr_intersector.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_ssr/src/snitch_ssr_addr_gen.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_ssr/src/snitch_ssr.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_ssr/src/snitch_ssr_streamer.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"
lappend search_path "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/mem_interface/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/reqrsp_interface/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_ssr/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/tcdm_interface/include"
lappend search_path "$ROOT/hw/vendor/pulp_platform_axi/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_cluster/src/snitch_data_mem.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_cluster/src/snitch_amo_shim.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_cluster/src/snitch_cluster_peripheral/snitch_cluster_peripheral_reg_pkg.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_cluster/src/snitch_cluster_peripheral/snitch_cluster_peripheral_reg_top.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_cluster/src/snitch_cluster_peripheral/snitch_cluster_peripheral.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_cluster/src/snitch_fpu.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_cluster/src/snitch_sequencer.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_cluster/src/snitch_tcdm_interconnect.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_cluster/src/snitch_barrier.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_cluster/src/snitch_fp_ss.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_cluster/src/snitch_shared_muldiv.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_cluster/src/snitch_cc.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_cluster/src/snitch_clkdiv2.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_cluster/src/snitch_hive.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_cluster/src/snitch_cluster.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"
lappend search_path "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/mem_interface/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/reqrsp_interface/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_ssr/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/tcdm_interface/include"
lappend search_path "$ROOT/hw/vendor/pulp_platform_axi/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snax_util/src/snax_csr_mux_demux.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snax_util/src/snax_acc_mux_demux.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snax_util/src/snax_intf_translator.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"
lappend search_path "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/mem_interface/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/reqrsp_interface/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_ssr/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/tcdm_interface/include"
lappend search_path "$ROOT/hw/vendor/pulp_platform_axi/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/target/snitch_cluster/generated/snax_versacore/VersaCore.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/target/snitch_cluster/generated/snax_versacore/snax_versacore_shell_wrapper.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/target/snitch_cluster/generated/snax_versacore/snax_versacore_csrman_CsrManager.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/target/snitch_cluster/generated/snax_versacore/snax_versacore_Streamer.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/target/snitch_cluster/generated/snax_versacore/snax_versacore_csrman_wrapper.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/target/snitch_cluster/generated/snax_versacore/snax_versacore_streamer_wrapper.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/target/snitch_cluster/generated/snax_versacore/snax_versacore_wrapper.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"
lappend search_path "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/mem_interface/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/reqrsp_interface/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_ssr/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/tcdm_interface/include"
lappend search_path "$ROOT/hw/vendor/pulp_platform_axi/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/target/snitch_cluster/generated/snax_versacore_cluster_xdma/snax_versacore_cluster_xdma.sv" \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/target/snitch_cluster/generated/snax_versacore_cluster_xdma/snax_versacore_cluster_xdma_wrapper.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"
lappend search_path "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/mem_interface/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/reqrsp_interface/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_ssr/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/tcdm_interface/include"
lappend search_path "$ROOT/hw/vendor/pulp_platform_axi/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/target/snitch_cluster/generated/snax_versacore_cluster_wrapper.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/apb-967c1a20a8f82c97/include"
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"
lappend search_path "$ROOT/.bender/git/checkouts/floo_noc-696cf2b1eb86a8f9/hw/include"
lappend search_path "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/mem_interface/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/reqrsp_interface/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_ssr/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/tcdm_interface/include"
lappend search_path "$ROOT/hw/spm_interface/include"
lappend search_path "$ROOT/hw/vendor/pulp_platform_axi/include"
lappend search_path "$ROOT/hw/vendor/pulp_platform_axi_tlb/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/hw/spm_interface/src/spm_interface.sv" \
        "$ROOT/hw/spm_interface/src/spm_rmw_adapter.sv" \
        "$ROOT/hw/spm_interface/src/spm_1p_adv_mem_wrapper.sv" \
        "$ROOT/hw/spm_interface/src/spm_1p_adv.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/apb-967c1a20a8f82c97/include"
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"
lappend search_path "$ROOT/.bender/git/checkouts/floo_noc-696cf2b1eb86a8f9/hw/include"
lappend search_path "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/mem_interface/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/reqrsp_interface/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_ssr/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/tcdm_interface/include"
lappend search_path "$ROOT/hw/spm_interface/include"
lappend search_path "$ROOT/hw/vendor/pulp_platform_axi/include"
lappend search_path "$ROOT/hw/vendor/pulp_platform_axi_tlb/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/target/rtl/src/quadrant_s1_ctrl/occamy_quadrant_s1_reg_pkg.sv" \
        "$ROOT/target/rtl/src/quadrant_s1_ctrl/occamy_quadrant_s1_reg_top.sv" \
        "$ROOT/target/rtl/src/soc_ctrl/occamy_soc_reg_pkg.sv" \
        "$ROOT/target/rtl/src/soc_ctrl/occamy_soc_reg_top.sv" \
        "$ROOT/target/rtl/src/soc_ctrl/occamy_soc_ctrl.sv" \
        "$ROOT/target/rtl/src/rv_plic/rv_plic_reg_pkg.sv" \
        "$ROOT/target/rtl/src/rv_plic/rv_plic_reg_top.sv" \
        "$ROOT/target/rtl/src/rv_plic/rv_plic.sv" \
        "$ROOT/target/rtl/src/clint/clint_reg_pkg.sv" \
        "$ROOT/target/rtl/src/clint/clint_reg_top.sv" \
        "$ROOT/target/rtl/src/clint/clint.sv" \
        "$ROOT/target/rtl/src/occamy_pkg.sv" \
        "$ROOT/target/rtl/src/occamy_quadrant_s1_ctrl.sv" \
        "$ROOT/target/rtl/src/occamy_quadrant_s1.sv" \
        "$ROOT/target/rtl/src/occamy_soc.sv" \
        "$ROOT/target/rtl/src/occamy_top.sv" \
        "$ROOT/target/rtl/src/occamy_cva6.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/apb-967c1a20a8f82c97/include"
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"
lappend search_path "$ROOT/.bender/git/checkouts/floo_noc-696cf2b1eb86a8f9/hw/include"
lappend search_path "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/mem_interface/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/reqrsp_interface/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_ssr/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/tcdm_interface/include"
lappend search_path "$ROOT/hw/spm_interface/include"
lappend search_path "$ROOT/hw/vendor/pulp_platform_axi/include"
lappend search_path "$ROOT/hw/vendor/pulp_platform_axi_tlb/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/target/rtl/bootrom/bootrom_chip/bootrom.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
lappend search_path "$ROOT/.bender/git/checkouts/apb-967c1a20a8f82c97/include"
lappend search_path "$ROOT/.bender/git/checkouts/common_cells-2daae953864f41bd/include"
lappend search_path "$ROOT/.bender/git/checkouts/floo_noc-696cf2b1eb86a8f9/hw/include"
lappend search_path "$ROOT/.bender/git/checkouts/register_interface-126aa152b770f2ca/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/mem_interface/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/reqrsp_interface/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/snitch_ssr/include"
lappend search_path "$ROOT/.bender/git/checkouts/snitch_cluster-9a64e6b3572c7a8d/hw/tcdm_interface/include"
lappend search_path "$ROOT/hw/spm_interface/include"
lappend search_path "$ROOT/hw/vendor/pulp_platform_axi/include"
lappend search_path "$ROOT/hw/vendor/pulp_platform_axi_tlb/include"

if {0 == [analyze -format sv \
    -define { \
        TARGET_CV64A6_IMAFDC_SV39 \
        TARGET_HEMAIA \
        TARGET_OCCAMY \
        TARGET_RTL \
        TARGET_SNAX_VERSACORE \
        TARGET_SNAX_VERSACORE_CLUSTER \
        TARGET_SNAX_VERSACORE_CLUSTER_XDMA \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
        TARGET_TSMC16 \
    } \
    [list \
        "$ROOT/hw/hemaia/hemaia_mem_system.sv" \
        "$ROOT/hw/hemaia/hemaia_clk_rst_controller/regs/hemaia_clk_rst_controller_reg_pkg.sv" \
        "$ROOT/hw/hemaia/hemaia_clk_rst_controller/regs/hemaia_clk_rst_controller_reg_top.sv" \
        "$ROOT/hw/hemaia/hemaia_clk_rst_controller/hemaia_clock_divider.sv" \
        "$ROOT/hw/hemaia/hemaia_clk_rst_controller/hemaia_reset_controller.sv" \
        "$ROOT/hw/hemaia/hemaia_clk_rst_controller/hemaia_clk_rst_controller.sv" \
        "$ROOT/target/rtl/src/occamy_chip.sv" \
    ]
]} {return 1}

set search_path $search_path_initial
