/* SPDX-License-Identifier: Apache-2.0 */
/* STM32H753 / STM32H743 Cortex-M7 Bare-Metal Startup with ITCM/DTCM Relocation (DR-08 Sec 1.2) */

  .syntax unified
  .cpu cortex-m7
  .fpu fpv5-d16
  .thumb

  .global g_flash_vectors
  .global g_itcm_vectors
  .global Reset_Handler
  .global Default_Handler

  .extern _estack
  .extern _litcm_vec
  .extern _sitcm_vec
  .extern _eitcm_vec
  .extern _litcm_text
  .extern _sitcm_text
  .extern _eitcm_text
  .extern _ldtcm_rodata
  .extern _sdtcm_rodata
  .extern _edtcm_rodata
  .extern _ldata
  .extern _sdata
  .extern _edata
  .extern _sbss
  .extern _ebss
  .extern __init_array_start
  .extern __init_array_end
  .extern main

/* Helper macro to declare weak IRQ handlers with mandatory .thumb_func + .type %function
 * (prevents GNU ld corrupt ITCM/Flash veneer opcode 0x04 F0 1F E5 under -O3). */
.macro WEAK_HANDLER name
  .weak \name
  .thumb_set \name, Default_Handler
.endm

  .section .text.Reset_Handler, "ax", %progbits
  .thumb_func
  .type Reset_Handler, %function
Reset_Handler:
  cpsid i
  ldr   r0, =_estack
  mov   sp, r0

  /* 1. Copy relocated vector table into ITCM @ 0x00000400 (above 1 KB NULL trap). */
  ldr   r0, =_sitcm_vec
  ldr   r1, =_eitcm_vec
  ldr   r2, =_litcm_vec
  bl    copy_words

  /* 2. Copy hot flight_core + math helpers into zero-wait-state ITCM. */
  ldr   r0, =_sitcm_text
  ldr   r1, =_eitcm_text
  ldr   r2, =_litcm_text
  bl    copy_words

  /* 3. Copy hot control constants into DTCM. */
  ldr   r0, =_sdtcm_rodata
  ldr   r1, =_edtcm_rodata
  ldr   r2, =_ldtcm_rodata
  bl    copy_words

  /* 4. Copy .data into DTCM. */
  ldr   r0, =_sdata
  ldr   r1, =_edata
  ldr   r2, =_ldata
  bl    copy_words

  /* 5. Zero .bss in DTCM. */
  ldr   r0, =_sbss
  ldr   r1, =_ebss
  movs  r2, #0
zero_bss_loop:
  cmp   r0, r1
  bhs   zero_bss_done
  str   r2, [r0], #4
  b     zero_bss_loop
zero_bss_done:

  /* 6. Synchronize TCM writes and point SCB->VTOR (0xE000ED08) to 0x00000400. */
  dsb   sy
  isb   sy
  ldr   r0, =0xE000ED08
  ldr   r1, =0x00000400
  str   r1, [r0]
  dsb   sy
  isb   sy

  /* 7. Enable CP10/CP11 full FPU access (SCB->CPACR @ 0xE000ED88) before C++ static ctors. */
  ldr   r0, =0xE000ED88
  ldr   r1, [r0]
  orr   r1, r1, #0x00F00000
  str   r1, [r0]
  dsb   sy
  isb   sy

  /* 8. Execute C++ static constructors (.init_array) before entering main(). */
  ldr   r4, =__init_array_start
  ldr   r5, =__init_array_end
init_array_loop:
  cmp   r4, r5
  bhs   init_array_done
  ldr   r0, [r4], #4
  blx   r0
  b     init_array_loop
init_array_done:

  cpsie i
  ldr   r0, =main
  blx   r0
hang_after_main:
  wfi
  b     hang_after_main
  .size Reset_Handler, .-Reset_Handler

  .section .text.copy_words, "ax", %progbits
  .thumb_func
  .type copy_words, %function
copy_words:
  cmp   r0, r1
  bhs   copy_words_done
  ldr   r3, [r2], #4
  str   r3, [r0], #4
  b     copy_words
copy_words_done:
  bx    lr
  .size copy_words, .-copy_words

  .section .itcm_text.Default_Handler, "ax", %progbits
  .thumb_func
  .type Default_Handler, %function
Default_Handler:
  b     Default_Handler
  .size Default_Handler, .-Default_Handler

  WEAK_HANDLER NMI_Handler
  WEAK_HANDLER HardFault_Handler
  WEAK_HANDLER MemManage_Handler
  WEAK_HANDLER BusFault_Handler
  WEAK_HANDLER UsageFault_Handler
  WEAK_HANDLER SVC_Handler
  WEAK_HANDLER DebugMon_Handler
  WEAK_HANDLER PendSV_Handler
  WEAK_HANDLER SysTick_Handler
  WEAK_HANDLER FDCAN1_IT0_IRQHandler
  WEAK_HANDLER FDCAN1_IT1_IRQHandler
  WEAK_HANDLER TIM2_IRQHandler
  WEAK_HANDLER USART3_IRQHandler
  WEAK_HANDLER OTG_FS_IRQHandler

/* Boot vector table in Flash at 0x08000000 (initial SP + Reset_Handler). */
  .section .flash_isr_vector, "a", %progbits
  .align 10
  .type g_flash_vectors, %object
g_flash_vectors:
  .word _estack
  .word Reset_Handler
  .word NMI_Handler
  .word HardFault_Handler
  .word MemManage_Handler
  .word BusFault_Handler
  .word UsageFault_Handler
  .word 0
  .word 0
  .word 0
  .word 0
  .word SVC_Handler
  .word DebugMon_Handler
  .word 0
  .word PendSV_Handler
  .word SysTick_Handler
  .rept 240
  .word Default_Handler
  .endr
  .size g_flash_vectors, .-g_flash_vectors

/* Relocated 1,024-byte NVIC vector table in ITCM at 0x00000400 (16 system + 240 IRQs). */
  .section .itcm_isr_vector, "a", %progbits
  .align 10
  .type g_itcm_vectors, %object
g_itcm_vectors:
  .word _estack
  .word Reset_Handler
  .word NMI_Handler
  .word HardFault_Handler
  .word MemManage_Handler
  .word BusFault_Handler
  .word UsageFault_Handler
  .word 0
  .word 0
  .word 0
  .word 0
  .word SVC_Handler
  .word DebugMon_Handler
  .word 0
  .word PendSV_Handler
  .word SysTick_Handler
  /* External IRQs 0..239 (STM32H753): FDCAN1_IT0=19, FDCAN1_IT1=21, TIM2=28, USART3=39, OTG_FS=101 */
  .rept 19
  .word Default_Handler
  .endr
  .word FDCAN1_IT0_IRQHandler
  .word Default_Handler
  .word FDCAN1_IT1_IRQHandler
  .rept 6
  .word Default_Handler
  .endr
  .word TIM2_IRQHandler
  .rept 10
  .word Default_Handler
  .endr
  .word USART3_IRQHandler
  .rept 61
  .word Default_Handler
  .endr
  .word OTG_FS_IRQHandler
  .rept 138
  .word Default_Handler
  .endr
  .size g_itcm_vectors, .-g_itcm_vectors
