/* Copyright 2020 ETH Zurich and University of Bologna. */
/* Solderpad Hardware License, Version 0.51, see LICENSE for details. */
/* SPDX-License-Identifier: SHL-0.51 */

/* The host image's layout. The same script links one image for every chiplet and, for a bingo workload with
   params per_chip_image, one image per chiplet (target/sw/host/apps/per_chip_image.py, which links each at its
   chiplet's own addresses and passes the padding below). The order keeps what another chiplet reaches, or what
   must not move, at one address in every chiplet's image whatever its code: the entry (the boot ROM jumps to the
   start of WIDE_SPM), the data (the workload's L3 symbols, the runtime's globals), the HTIF words and the BSS
   come first; the code and the read-only data follow in one region, which a per-chiplet build pads to the
   largest chiplet's (__bingo_var_size), so the device binary (linked at snitch_main) and the L3 heap
   (bingo_api.c derives another chiplet's heap from its own __l3_heap_start) sit at one address too. */

ENTRY(_start)

MEMORY
{
    NARROW_SPM (rwxa) : ORIGIN = ${spm_narrow["address"]}, LENGTH = ${spm_narrow["length"]}
    WIDE_SPM (rwxa) : ORIGIN = ${spm_wide["address"]}, LENGTH = ${spm_wide["length"]}
}

/* the padded size of the code region; 0 (no padding) unless the link defines it (per_chip_image.py) */
PROVIDE(__bingo_var_size = 0);

SECTIONS
{
  /* The entry, then the data */
  .appl :
  {
    /* __stack_pointer$  = . + 0x70000; */
    /* __global_pointer$ = . + 0x7f0; */
    __return_pointer$ = ORIGIN(WIDE_SPM) + LENGTH(WIDE_SPM) - 4;
    *(.text.startup)
    . = ALIGN(64);
    __bingo_shared_data_start = .;
    __SDATA_BEGIN__ = .;
    *(.sdata .sdata.* .gnu.linkonce.s.*)
    *(.data .data.*)
    __bingo_shared_data_end = .;
  } > WIDE_SPM

  /* HTIF section for FESVR */
  .htif : { *(.htif) } > WIDE_SPM

  .bss (NOLOAD) :
  {
    . = ALIGN(8);
    __bss_start = . ;
    *(.sbss .sbss.*)
    *(.bss .bss.*)
    *(COMMON)
    . = ALIGN(8);
    __bss_end = . ;
  } > WIDE_SPM

  /* The code and the read-only data: what differs between chiplets in a per-chiplet build (with each chiplet's
     own task arguments, .bingo_chip.data), padded there to one size */
  .code :
  {
    . = ALIGN(64);
    __bingo_var_start = .;
    *(.text .text.*)
    *(.srodata.cst16) *(.srodata.cst8) *(.srodata.cst4) *(.srodata.cst2) *(.srodata .srodata.*)
    *(.rodata .rodata.*)
    *(.bingo_chip.data)
    __bingo_var_end = .;
    . = MAX(., __bingo_var_start + __bingo_var_size);
  } > WIDE_SPM

  .wide_spm :
  {
    __wide_spm_start = ORIGIN(WIDE_SPM);
    __wide_spm_end = ORIGIN(WIDE_SPM) + LENGTH(WIDE_SPM);
  } > WIDE_SPM

  .narrow_spm :
  {
    __narrow_spm_start = ORIGIN(NARROW_SPM);
    __narrow_spm_end = ORIGIN(NARROW_SPM) + LENGTH(NARROW_SPM);
  } > WIDE_SPM

  . = ALIGN(8);
  __host_end = .;

  .devicebin : { *(.devicebin) } > WIDE_SPM
  .l3_heap : {
    __l3_heap_start = .;
  } > WIDE_SPM

  /* Discard sections */
  /DISCARD/ : { *(.riscv.attributes) *(.comment) }
}
