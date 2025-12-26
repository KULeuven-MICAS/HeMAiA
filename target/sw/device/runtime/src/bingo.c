// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>

extern void bingo_wait_worker_wfi();
extern void bingo_wake_workers();
extern void bingo_worker_wfi(uint32_t cluster_core_idx);
extern void bingo_print_status();
extern uint32_t bingo_get_workers_in_loop();
extern uint32_t bingo_get_workers_in_wfi();
extern void bingo_offload_init();
extern void bingo_offload_exit();
extern void bingo_offload_event_loop(uint32_t cluster_core_idx);
extern void bingo_offload_dispatch(uint32_t (*offloadFn)(uint32_t), uint32_t offloadArgs);
extern uint32_t bingo_offload_manager();
extern uint32_t bingo_hw_offload_manager();