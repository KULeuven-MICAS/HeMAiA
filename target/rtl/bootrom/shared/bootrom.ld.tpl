/* Copyright 2024 KU Leuven. */


ENTRY(_start)

MEMORY
{
  bootrom (rx)        : ORIGIN = 0x01000000, LENGTH = 16K
  narrow_spm (rwx)    : ORIGIN = 0x70000000, LENGTH = 32K
  wide_spm (rwx)      : ORIGIN = 0x80000000, LENGTH = 512K
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
  __soc_ctrl_scratch0     = 0x02000014;
  __soc_ctrl_scratch1     = 0x02000018;
  __soc_ctrl_scratch2     = 0x0200001c;
}
