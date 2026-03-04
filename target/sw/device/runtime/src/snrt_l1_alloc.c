// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>

// We put the snrt_l1_allocator in the .cbss section for fast access from the cluster.
// According to the base.ld
/* Cluster Local Storage sections */
//   .cdata    :
//   {
//     __cdata_start = .;
//     *(.cdata .cdata.*)
//     __cdata_end = .;
//   } >L3
//   .cbss :
//   {
//     __cbss_start = .;
//     *(.cbss .cbss.*)
//     __cbss_end = .;
//   } >L3

// __attribute__((section(".cbss"))) snrt_l1_allocator_t l1_allocator;

extern uint32_t snrt_l1_malloc(uint32_t size);
extern void snrt_l1_free(uint32_t addr);
extern void snrt_l1_malloc_init();