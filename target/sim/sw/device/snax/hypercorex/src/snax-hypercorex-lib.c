// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

//-------------------------------
// Author: Ryan Antonio <ryan.antonio@esat.kuleuven.be>
//
// Library: Functions for Setting Hypecorex CSRs
//
// This pre-built library contains functions to set
// the HyperCoreX accelerator's CSRs.
//-------------------------------

#include "snax-hypercorex-lib.h"
#include "streamer_csr_addr_map.h"

//-------------------------------
// Streamer functions
//-------------------------------
void hypercorex_set_streamer_lowdim_a(
    uint32_t base_ptr_low, uint32_t base_ptr_high, uint32_t spat_stride,
    uint32_t loop_bound_0, uint32_t loop_bound_1, uint32_t temp_stride_0,
    uint32_t temp_stride_1) {
    csrw_ss(BASE_PTR_READER_0_LOW, base_ptr_low);
    csrw_ss(BASE_PTR_READER_0_HIGH, base_ptr_high);
    csrw_ss(S_STRIDE_READER_0_0, spat_stride);
    csrw_ss(T_BOUND_READER_0_0, loop_bound_0);
    csrw_ss(T_BOUND_READER_0_1, loop_bound_1);
    csrw_ss(T_STRIDE_READER_0_0, temp_stride_0);
    csrw_ss(T_STRIDE_READER_0_1, temp_stride_1);
    return;
};

void hypercorex_set_streamer_lowdim_b(
    uint32_t base_ptr_low, uint32_t base_ptr_high, uint32_t spat_stride,
    uint32_t loop_bound_0, uint32_t loop_bound_1, uint32_t temp_stride_0,
    uint32_t temp_stride_1) {
    csrw_ss(BASE_PTR_READER_1_LOW, base_ptr_low);
    csrw_ss(BASE_PTR_READER_1_HIGH, base_ptr_high);
    csrw_ss(S_STRIDE_READER_1_0, spat_stride);
    csrw_ss(T_BOUND_READER_1_0, loop_bound_0);
    csrw_ss(T_BOUND_READER_1_1, loop_bound_1);
    csrw_ss(T_STRIDE_READER_1_0, temp_stride_0);
    csrw_ss(T_STRIDE_READER_1_1, temp_stride_1);
    return;
};

void hypercorex_set_streamer_highdim_a(
    uint32_t base_ptr_low, uint32_t base_ptr_high, uint32_t spat_stride,
    uint32_t loop_bound_0, uint32_t loop_bound_1, uint32_t temp_stride_0,
    uint32_t temp_stride_1) {
    csrw_ss(BASE_PTR_READER_2_LOW, base_ptr_low);
    csrw_ss(BASE_PTR_READER_2_HIGH, base_ptr_high);
    csrw_ss(S_STRIDE_READER_2_0, spat_stride);
    csrw_ss(T_BOUND_READER_2_0, loop_bound_0);
    csrw_ss(T_BOUND_READER_2_1, loop_bound_1);
    csrw_ss(T_STRIDE_READER_2_0, temp_stride_0);
    csrw_ss(T_STRIDE_READER_2_1, temp_stride_1);
    return;
};

void hypercorex_set_streamer_highdim_b(
    uint32_t base_ptr_low, uint32_t base_ptr_high, uint32_t spat_stride,
    uint32_t loop_bound_0, uint32_t loop_bound_1, uint32_t temp_stride_0,
    uint32_t temp_stride_1) {
    csrw_ss(BASE_PTR_READER_3_LOW, base_ptr_low);
    csrw_ss(BASE_PTR_READER_3_HIGH, base_ptr_high);
    csrw_ss(S_STRIDE_READER_3_0, spat_stride);
    csrw_ss(T_BOUND_READER_3_0, loop_bound_0);
    csrw_ss(T_BOUND_READER_3_1, loop_bound_1);
    csrw_ss(T_STRIDE_READER_3_0, temp_stride_0);
    csrw_ss(T_STRIDE_READER_3_1, temp_stride_1);
    return;
};

void hypercorex_set_streamer_highdim_am(
    uint32_t base_ptr_low, uint32_t base_ptr_high, uint32_t spat_stride,
    uint32_t loop_bound_0, uint32_t loop_bound_1, uint32_t temp_stride_0,
    uint32_t temp_stride_1) {
    csrw_ss(BASE_PTR_READER_4_LOW, base_ptr_low);
    csrw_ss(BASE_PTR_READER_4_HIGH, base_ptr_high);
    csrw_ss(S_STRIDE_READER_4_0, spat_stride);
    csrw_ss(T_BOUND_READER_4_0, loop_bound_0);
    csrw_ss(T_BOUND_READER_4_1, loop_bound_1);
    csrw_ss(T_STRIDE_READER_4_0, temp_stride_0);
    csrw_ss(T_STRIDE_READER_4_1, temp_stride_1);
    return;
};

void hypercorex_set_streamer_lowdim_predict(
    uint32_t base_ptr_low, uint32_t base_ptr_high, uint32_t spat_stride,
    uint32_t loop_bound_0, uint32_t loop_bound_1, uint32_t temp_stride_0,
    uint32_t temp_stride_1) {
    csrw_ss(BASE_PTR_WRITER_0_LOW, base_ptr_low);
    csrw_ss(BASE_PTR_WRITER_0_HIGH, base_ptr_high);
    csrw_ss(S_STRIDE_WRITER_0_0, spat_stride);
    csrw_ss(T_BOUND_WRITER_0_0, loop_bound_0);
    csrw_ss(T_BOUND_WRITER_0_1, loop_bound_1);
    csrw_ss(T_STRIDE_WRITER_0_0, temp_stride_0);
    csrw_ss(T_STRIDE_WRITER_0_1, temp_stride_1);
    return;
};

void hypercorex_set_streamer_highdim_qhv(
    uint32_t base_ptr_low, uint32_t base_ptr_high, uint32_t spat_stride,
    uint32_t loop_bound_0, uint32_t loop_bound_1, uint32_t temp_stride_0,
    uint32_t temp_stride_1) {
    csrw_ss(BASE_PTR_WRITER_1_LOW, base_ptr_low);
    csrw_ss(BASE_PTR_WRITER_1_HIGH, base_ptr_high);
    csrw_ss(S_STRIDE_WRITER_1_0, spat_stride);
    csrw_ss(T_BOUND_WRITER_1_0, loop_bound_0);
    csrw_ss(T_BOUND_WRITER_1_1, loop_bound_1);
    csrw_ss(T_STRIDE_WRITER_1_0, temp_stride_0);
    csrw_ss(T_STRIDE_WRITER_1_1, temp_stride_1);
    return;
};

void hypercorex_start_streamer(void) {
    csrw_ss(STREAMER_START_CSR, 1);
    return;
};

uint32_t hypercorex_is_streamer_busy(void) {
    return csrr_ss(STREAMER_BUSY_CSR);
};

uint32_t hypercorex_read_perf_counter(void) {
    return csrr_ss(STREAMER_PERFORMANCE_COUNTER_CSR);
};

//-------------------------------
// Instruction loading functions
//-------------------------------

void hypercorex_load_inst(uint32_t inst_size, uint32_t start_addr,
                          uint32_t* inst_list) {
    // First enable instruction write mode
    csrw_ss(HYPERCOREX_INST_CTRL_REG_ADDR, 0x00000001);

    // Set starting address
    csrw_ss(HYPERCOREX_INST_WRITE_ADDR_REG_ADDR, start_addr);

    // Load all instructions
    for (uint32_t i = 0; i < inst_size; i++) {
        csrw_ss(HYPERCOREX_INST_WRITE_DATA_REG_ADDR, inst_list[i]);
    };

    // Disable instruction write mode and reset PC
    csrw_ss(HYPERCOREX_INST_CTRL_REG_ADDR, 0x00000004);

    return;
};

//-------------------------------
// Instruction loop control functions
//
// These isntructions take in 7 bits per configuration
// and packs them into one 32-bit register
// Upper MSBs are tied to 0s
//
// Note: This is a parameter that changes
// depending on how large the instruction memory is
//-------------------------------

void hypercorex_set_inst_loop_jump_addr(uint8_t config1, uint8_t config2,
                                        uint8_t config3) {
    uint32_t config =
        ((config3 & 0x7f) << 14) | ((config2 & 0x7f) << 7) | (config1 & 0x7f);

    csrw_ss(HYPERCOREX_INST_LOOP_JUMP_ADDR_REG_ADDR, config);
    return;
};

void hypercorex_set_inst_loop_end_addr(uint8_t config1, uint8_t config2,
                                       uint8_t config3) {
    uint32_t config =
        ((config3 & 0x7f) << 14) | ((config2 & 0x7f) << 7) | (config1 & 0x7f);

    csrw_ss(HYPERCOREX_INST_LOOP_END_ADDR_REG_ADDR, config);
    return;
};

void hypercorex_set_inst_loop_count(uint32_t config1, uint32_t config2,
                                    uint32_t config3) {
    uint32_t config = ((config3 & 0x3ff) << 20) | ((config2 & 0x3ff) << 10) |
                      (config1 & 0x3ff);

    csrw_ss(HYPERCOREX_INST_LOOP_COUNT_REG_ADDR, config);
    return;
};

void hypercorex_start_core(void) {
    csrw_ss(HYPERCOREX_CORE_SET_REG_ADDR, 1);
    return;
};

uint32_t hypercorex_is_core_busy(void) {
    return csrr_ss(HYPERCOREX_CORE_SET_REG_ADDR);
};

void hypercorex_pack_512b(
    uint32_t vec_num,
    uint32_t *tcdm_super_bank_1,
    uint32_t *tcdm_super_bank_3
){
    int32_t main32b;
    int8_t sub8b;
    uint32_t sub32b_hv;
    uint32_t hv_32b_cnt;
    // num_hv_rows is how many HV in int8 needs to be processed
    for (uint32_t num_hv_rows = 0; num_hv_rows < vec_num; num_hv_rows++){
        // num_super_bank_rows is the number of super bank rows for 1 HV
        hv_32b_cnt = 1;
        for (uint32_t num_super_bank_rows = 0; num_super_bank_rows < 8; num_super_bank_rows++){
            // super_bank_sub_32b_idx is the index of an entire 32b data in onw super bank row
            // there are 8 banks in 1 super bank. the first 4 banks (3:0) gives one 32b chunk.
            // the 2nd 4 banks (7:4) gives the second 32b chunk
            for (uint32_t super_bank_sub_32b_idx = 0; super_bank_sub_32b_idx < 2; super_bank_sub_32b_idx++){
                // Initialize extraction of sub32b HV
                uint32_t sub32b_hv = 0;
                // bank_sub_32b_idx is the bank number of the first 4 banks.
                // Note that the first 4 banks have 8 32-bit values packed together
                for (uint32_t bank_sub_32b_idx = 0; bank_sub_32b_idx < 8; bank_sub_32b_idx++){
                    // Extract the 1st 32b data
                    main32b = *(tcdm_super_bank_1 +         // Base address
                                num_hv_rows*512 +           // HV stride for int32
                                num_super_bank_rows*64 +    // Row per-super bank stride
                                super_bank_sub_32b_idx*8 +  // Sub-super bank stride
                                bank_sub_32b_idx);          // Per bank stride
                    // Per 32b data there are 4 int8 data
                    // This is where we combine the bits to make 1b data
                    for (uint32_t sub_int8b_idx = 0; sub_int8b_idx < 4; sub_int8b_idx++){
                        sub8b = (main32b >> (sub_int8b_idx * 8)) & 0x000000FF;
                        if (sub8b >= 0){
                            sub32b_hv = (sub32b_hv << 1) | 0x00000001;
                        } else {
                            sub32b_hv = (sub32b_hv << 1) | 0x00000000;
                        }
                    }
                }
                // Save it unto the proper banks
                // Note that we have to reverse the order
                // So we start at the end and fill it backwards
                // As the hypercorex uses big endian
                *(tcdm_super_bank_3 + num_hv_rows*64 - (1 * hv_32b_cnt)) = sub32b_hv;
                hv_32b_cnt++;
            }
        }
    }
    return;
};

//-------------------------------
// Data slicer functions
//
// These are used for the data slicing mechanism
// of the Hypercorex. Where packed data can be sliced
// into smaller chunks for processing.
//-------------------------------

void hypercorex_set_data_slice_ctrl(uint8_t slice_ctrl_a, uint8_t slice_ctrl_b,
                                    uint8_t slice_src_a, uint8_t slice_src_b) {
    uint32_t config = (slice_src_b & 0x1) << 5 | (slice_src_a & 0x1) << 4 |
                      (slice_ctrl_b & 0x3) << 2 | (slice_ctrl_a & 0x3);

    csrw_ss(HYPERCOREX_DATA_SLICE_CTRL, config);
    return;
}

void hypercorex_set_data_slice_num_elem_a(uint32_t num_elem) {
    csrw_ss(HYPERCOREX_DATA_SLICE_NUM_ELEM_A, num_elem);
    return;
}

void hypercorex_set_data_slice_num_elem_b(uint32_t num_elem) {
    csrw_ss(HYPERCOREX_DATA_SLICE_NUM_ELEM_B, num_elem);
    return;
}

//-------------------------------
// Auto-updater functions
//
// These are used for autofetching data
// from the the data streamer of the Hypercorex.
// Useful for automatic counting for IM
//-------------------------------

void hypercorex_set_auto_counter_start_a(uint32_t start_counter) {
    csrw_ss(HYPERCOREX_AUTO_COUNTER_START_A, start_counter);
    return;
}

void hypercorex_set_auto_counter_start_b(uint32_t start_counter) {
    csrw_ss(HYPERCOREX_AUTO_COUNTER_START_B, start_counter);
    return;
}

void hypercorex_set_auto_counter_num_a(uint32_t num_counter) {
    csrw_ss(HYPERCOREX_AUTO_COUNTER_NUM_A, num_counter);
    return;
}

void hypercorex_set_auto_counter_num_b(uint32_t num_counter) {
    csrw_ss(HYPERCOREX_AUTO_COUNTER_NUM_B, num_counter);
    return;
}