/* Copyright 2020 ETH Zurich and University of Bologna. */
/* Solderpad Hardware License, Version 0.51, see LICENSE for details. */
/* SPDX-License-Identifier: SHL-0.51 */

ENTRY(_start)

MEMORY
{
    DRAM (rwxa) : ORIGIN = 0x80000000, LENGTH = 0x100000
}

SECTIONS
{
  /* The program code and data goes into DRAM */
  .appl :
  {
    /* __stack_pointer$  = . + 0x70000; */
    /* __global_pointer$ = . + 0x7f0; */
    
    __return_pointer$ = ORIGIN(DRAM) + LENGTH(DRAM) - 4;
    *(.text.startup)
    *(.text .text.*)
    __SDATA_BEGIN__ = .;
    *(.sdata)
    *(.srodata.cst16) *(.srodata.cst8) *(.srodata.cst4) *(.srodata.cst2) *(.srodata .srodata.*)
    *(.sdata .sdata.* .gnu.linkonce.s.*)
    *(.data)
    *(.rodata .rodata.*)
  } > DRAM

  /* HTIF section for FESVR */
  .htif : { *(.htif) } > DRAM

  .bss (NOLOAD) :
  {
    . = ALIGN(8);
    __bss_start = . ;
    *(.sbss .sbss.*)
    *(.bss .bss.*)
    *(COMMON)
    . = ALIGN(8);
    __bss_end = . ;
  } > DRAM

  .wide_spm :
  {
    . = ALIGN(8);
    __wide_spm_start = . ;
    *(.wide_spm)
    . = ALIGN(8);
    __wide_spm_end = . ;
  } > DRAM

  __end = .;

  .devicebin : { *(.devicebin) } > DRAM

  /* Discard sections */
  /DISCARD/ : { *(.riscv.attributes) *(.comment) }
}
