// Copyright 2024 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// JYIN

#include "streamer_csr_addr_map.h"
#include "snax-cgra-lib.h"

#include "snrt.h"
#include "stdint.h"
#include "data.h"

int main() {
	if (snrt_cluster_idx() == 0) {
		// Set err value for checking
		int err = 0;

		const char* stage_helper[7] = {
			"CTRL_REG",
			"CONFIG",
			"EXE_INI",
			"COMP_S1",
			"COMP_S2",
			"COMP_S3",
			"COMP_S4"
		};
		int perf_counter[2];
		int perf_counter_tmp;
		int fsafe_counter, err_counter;

		int32_t *local_config_data;
		int16_t *local_d16;

		local_config_data = (int32_t *)(snrt_l1_next() + delta_config_data);
		local_d16 = (int16_t *)(snrt_l1_next() + delta_store_data);

		// Using DMA only
		if (snrt_is_dm_core()) {
			snrt_dma_start_1d(local_config_data, CONFIG,
							CONFIG_SIZE * sizeof(uint32_t));

			snrt_dma_wait_all();
		}

		snrt_cluster_hw_barrier();

		local_config_data = (int32_t *)(snrt_l1_next() + delta_comp_data);
		// Using DMA only
		if (snrt_is_dm_core()) {
			snrt_dma_start_1d(local_config_data, COMP_DATA,
							COMP_DATA_SIZE * sizeof(uint32_t));

			snrt_dma_wait_all();
		}
		snrt_cluster_hw_barrier();

		// testing csr -> cgra
		if (snrt_is_compute_core()) {
			printf ("hello world!\n\n");

			launch_cgra_0(delta_config_data, delta_comp_data, delta_store_data);

			fsafe_counter = 0;
			while(csrr_ss(STREAMER_BUSY_CSR) != 0 || csrr_ss(CGRA_CSR_RO_ADDR_BASE+8) < 655370){
				printf ("system busy ...\n"); 
				if (csrr_ss(CGRA_CSR_RO_ADDR_BASE+8) >= 655370 && fsafe_counter >= 42) {
					printf ("--> Break because CGRA is done. Check the Streamer IO amount settings...\n"); 
					break;
				}
				fsafe_counter++;
			}
			printf ("ckpt hit!\n\n"); 
			
			
			printf("ckpt @ CGRA stage [%2d]\n", csrr_ss(CGRA_CSR_RO_ADDR_BASE+8) >> 16);
			printf("\tCGRA cc timestamp = %d\n", csrr_ss(CGRA_CSR_RO_ADDR_BASE));
			for(int i = 1; i < 7; i++) {
				perf_counter[0] = csrr_ss(CGRA_CSR_RO_ADDR_BASE+i);
				perf_counter[1] = csrr_ss(CGRA_CSR_RO_ADDR_BASE+i+1);
				perf_counter_tmp = perf_counter[1] < perf_counter[0] ? 0 : perf_counter[1] - perf_counter[0];
				printf("\tElapsed cc @ %s = %d\n", stage_helper[i], perf_counter_tmp);
			}
			printf("\tCGRA cc timestamp = %d\n\n", csrr_ss(CGRA_CSR_RO_ADDR_BASE));

			
			printf("Check result:\n");
			err_counter = 0;
			for(int i = 0; i < 48; i++) {
				if (local_d16[i*4] != COMP_DATA_REF[i]) err_counter++;
			}
			printf("\tErrors: %d\n", err_counter);
			return err_counter;
		} else return 0;
	} else return 0;
}
