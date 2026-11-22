/* SPDX-License-Identifier: Apache-2.0 */
/* VertiRTA Zero-Wait-State ITCM/DTCM Linker Script for STM32H753 / NUCLEO-H743ZI2 (DR-08 Sec 1.2) */

ENTRY(Reset_Handler)

MEMORY
{
  ITCM     (rx)  : ORIGIN = 0x00000000, LENGTH = 64K
  FLASH    (rx)  : ORIGIN = 0x08000000, LENGTH = 2048K
  DTCM     (rw)  : ORIGIN = 0x20000000, LENGTH = 128K
  AXI_SRAM (rw)  : ORIGIN = 0x24000000, LENGTH = 512K
  D2_SRAM1 (rw)  : ORIGIN = 0x30000000, LENGTH = 128K
}

_estack = ORIGIN(DTCM) + LENGTH(DTCM);

SECTIONS
{
  /* Cold reset vector in internal Flash at 0x08000000 (bootstrapped before ITCM relocation). */
  .flash_isr_vector 0x08000000 :
  {
    . = ALIGN(1024);
    KEEP(*(.flash_isr_vector))
    . = ALIGN(1024);
  } > FLASH

  /* Bottom 1,024 B of ITCM reserved for MPU Region R3 AP_NONE/XN=1 hardware NULL-pointer trap. */
  .itcm_null_guard 0x00000000 (NOLOAD) :
  {
    . = . + 1024;
  } > ITCM

  /* Relocated Cortex-M7 NVIC vector table at 1,024-byte aligned VMA 0x00000400 (SCB->VTOR). */
  _litcm_vec = LOADADDR(.itcm_isr_vector);
  .itcm_isr_vector 0x00000400 :
  {
    . = ALIGN(1024);
    _sitcm_vec = .;
    KEEP(*(.itcm_isr_vector))
    . = ALIGN(4);
    _eitcm_vec = .;
  } > ITCM AT> FLASH

  /* Zero-wait-state hot 400 Hz control archive + libgcc/libm helpers + veneers in 64 KB ITCM. */
  _litcm_text = LOADADDR(.itcm_text);
  .itcm_text :
  {
    . = ALIGN(4);
    _sitcm_text = .;
    *libvertirta_flight_core.a:*(.text .text.*)
    *(.itcm_text .itcm_text.*)
    *libgcc.a:*(.text .text.*)
    *libm.a:*(.text .text.*)
    *(.glue_7 .glue_7t .vfp11_veneer .v4_bx)
    . = ALIGN(4);
    _eitcm_text = .;
  } > ITCM AT> FLASH

  ASSERT((_eitcm_text <= (ORIGIN(ITCM) + LENGTH(ITCM))), "ERROR: .itcm_text exceeded 64 KB ITCM budget")

  /* Cold startup, diagnostics, and UART/FDCAN harness code in Flash. */
  .text :
  {
    . = ALIGN(4);
    *(.text .text.*)
    *(EXCLUDE_FILE(*libvertirta_flight_core.a) .rodata .rodata.*)
    . = ALIGN(4);
    _etext = .;
  } > FLASH

  .ARM.exidx :
  {
    *(.ARM.exidx* .gnu.linkonce.armexidx.*)
  } > FLASH

  /* C++ static constructor table invoked by Reset_Handler after TCM relocation & FPU enable. */
  .init_array :
  {
    . = ALIGN(4);
    PROVIDE_HIDDEN(__init_array_start = .);
    KEEP(*(SORT(.init_array.*)))
    KEEP(*(.init_array*))
    PROVIDE_HIDDEN(__init_array_end = .);
  } > FLASH

  /* Bottom 256 B of DTCM reserved for MPU Region R5 AP_NONE/XN=1 stack-overflow MemManage trap. */
  .dtcm_stack_guard 0x20000000 (NOLOAD) :
  {
    . = . + 256;
  } > DTCM

  /* Hot read-only control tables relocated into zero-wait-state DTCM. */
  _ldtcm_rodata = LOADADDR(.dtcm_rodata);
  .dtcm_rodata :
  {
    . = ALIGN(8);
    _sdtcm_rodata = .;
    *libvertirta_flight_core.a:*(.rodata .rodata.*)
    *(.dtcm_rodata .dtcm_rodata.*)
    . = ALIGN(8);
    _edtcm_rodata = .;
  } > DTCM AT> FLASH

  /* Initialized data in DTCM. */
  _ldata = LOADADDR(.data);
  .data :
  {
    . = ALIGN(8);
    _sdata = .;
    *(.data .data.*)
    . = ALIGN(8);
    _edata = .;
  } > DTCM AT> FLASH

  /* Zero-initialized BSS in DTCM (includes BSP static state; flight_core archive remains 0 B). */
  .bss (NOLOAD) :
  {
    . = ALIGN(8);
    _sbss = .;
    *(.bss .bss.*)
    *(COMMON)
    . = ALIGN(8);
    _ebss = .;
  } > DTCM

  ASSERT(((_estack - _ebss) >= 16384), "ERROR: Less than 16 KB descending stack remaining in DTCM")

  /* D1 AXI SRAM (512 KB, WB-WA S=0): holds 10,000-sample DWT->CYCCNT raw profiling ring (40 KB). */
  .axi_sram (NOLOAD) :
  {
    . = ALIGN(32);
    _saxi_sram = .;
    *(.axi_sram .axi_sram.*)
    . = ALIGN(32);
    _eaxi_sram = .;
  } > AXI_SRAM

  /* D2 SRAM1 (MPU Region R7 Normal Non-Cacheable 32 KB): DMA descriptors & TinyUSB DWC2 buffers. */
  .dma_nocache (NOLOAD) :
  {
    . = ALIGN(32);
    _sdma_nocache = .;
    *(.dma_nocache .dma_nocache.*)
    . = ALIGN(32);
    _edma_nocache = .;
  } > D2_SRAM1

  ASSERT(((_edma_nocache - _sdma_nocache) <= 32768), "ERROR: .dma_nocache exceeded 32 KB MPU R7 window")

  /DISCARD/ :
  {
    *(.eh_frame*)
    *(.note.GNU-stack)
  }
}
