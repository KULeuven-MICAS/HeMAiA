/* Copyright 2024 KU Leuven. */
/* Generated from bootrom.ld.tpl by util/gen_bootrom_ld.py -- do not edit. */


ENTRY(_start)

/* Regions come from the generated platform headers: BOOTROM_BASE_ADDR and
   SPM_NARROW_BASE_ADDR in occamy_base_addr.h, BOOTROM_SIZE and NARROW_SPM_SIZE in
   occamy_memory_map.h. Sizes are configuration dependent, so they must not be
   hardcoded. */
MEMORY
{
  bootrom (rx)        : ORIGIN = ${f"0x{bootrom_base:08x}"}, LENGTH = ${bootrom_size}
  narrow_spm (rwx)    : ORIGIN = ${f"0x{narrow_spm_base:08x}"}, LENGTH = ${narrow_spm_size}
}

SECTIONS
{
  .text : { *(.text._start) *(.text) } > bootrom
	.rodata : { *(.rodata) *(.rodata*) } > bootrom
	.srodata : { *(.srodata) *(.srodata*) } > bootrom
	.data : { *(.data) *(.sdata)  } > narrow_spm
	.bss : { *(.bss) } > narrow_spm

	/* .misc : { *(*) } > bootrom */
  /* /DISCARD/ : { *(.riscv.attributes) [reduce binary size] *(.comment) [reduce binary size] *(.debug*) [reduce binary size] *(.data) [no initialized memory] *(.sdata) [no initialized memory] } */

  /* Global and stack pointer */
  __global_pointer$       = ADDR(.rodata) + SIZEOF(.rodata) / 2;
  __stack_pointer$        = ORIGIN(narrow_spm) + LENGTH(narrow_spm) - 8;
  /* SoC Ctrl Reg for Snitch Function Calls */
  /* scratch0 -> snitch_main function ptr */
  /* scartch1 -> comm buffer ptr          */
  /* scartch2 -> snitch exit code val     */
  /* Derived from SOC_CTRL_BASE_ADDR in occamy_base_addr.h and
     OCCAMY_SOC_SCRATCH_n_REG_OFFSET in occamy_soc_ctrl.h, the same generated
     headers the host software uses, so the two cannot drift apart. The boot ROM
     reads the entry point from scratch0 in assembly, which is why these have to
     be linker symbols rather than C macros. */
% for i, addr in enumerate(soc_ctrl_scratch):
  __soc_ctrl_scratch${i}     = ${f"0x{addr:08x}"};
% endfor
}
