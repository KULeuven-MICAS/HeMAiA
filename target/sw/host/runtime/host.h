// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// #include "host.h"
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "chip_id.h"
#include "occamy.h"
#include "sys_dma.h"
// HeMAiA specific peripherals
#include "uart.c"
#include "hemaia_clk_rst_controller.h"
#include "hemaia_d2d_link.h"
#include "hemaia-xdma-lib.h"
#include "mailbox.h"
#include "io.h"
#include "heterogeneous_runtime.h"
// Host kernel lib
#include "host_kernel_lib.h"
extern uint64_t __narrow_spm_start;
extern uint64_t __narrow_spm_end;
extern uint64_t __wide_spm_start;
extern uint64_t __wide_spm_end;
// Handle multireg degeneration to single register
#if OCCAMY_SOC_SCRATCH_MULTIREG_COUNT == 1
#define OCCAMY_SOC_SCRATCH_0_REG_OFFSET OCCAMY_SOC_SCRATCH_0_REG_OFFSET
#endif
#if OCCAMY_SOC_SCRATCH_MULTIREG_COUNT == 1
#define OCCAMY_SOC_SCRATCH_0_REG_OFFSET OCCAMY_SOC_SCRATCH_REG_OFFSET
#endif

#define ARRAY_ELEM_COUNT(a) (sizeof(a)/sizeof((a)[0]))
#define ARRAY_SIZE_BYTES(a) (sizeof(a))
//===============================================================
// SNAX LIB Symbol Tab
//===============================================================

// symtab data structure
#define SNAX_LIB_NAME_MAX_LEN 64
typedef struct __attribute__((packed)){
    char name[SNAX_LIB_NAME_MAX_LEN];      // function name
    uint32_t addr;                         // function addr
} snax_symbol_t;
#define SNAX_SYMTAB_END_FN_NAME "SYMTAB_END"
#define SNAX_SYMTAB_END_FN_ADDR (uint32_t)(0xBAADF00D)

// Busy wait the ready signal
void check_kernel_tab_ready(){
    uint64_t symtab_ready_addr = (uint64_t)soc_ctrl_kernel_tab_scratch_addr(0);
    uint64_t local_symtab_ready_addr = chiplet_addr_transform(symtab_ready_addr);
    // printf("Chip(%x, %x): [Host] Kernel tab ready addr: 0x%lx\r\n", get_current_chip_loc_x(), get_current_chip_loc_y(), local_symtab_ready_addr);
    while(readw(local_symtab_ready_addr)!=1){

    }
}
// Get the device function
uint32_t get_device_function(const char *name) {
    uint64_t symtab_start_addr_ptr = (uint64_t)soc_ctrl_kernel_tab_scratch_addr(1);
    uint64_t symtab_end_addr_ptr   = (uint64_t)soc_ctrl_kernel_tab_scratch_addr(2);
    uint64_t local_symtab_start_addr_ptr = chiplet_addr_transform(symtab_start_addr_ptr);
    uint64_t local_symtab_end_addr_ptr   = chiplet_addr_transform(symtab_end_addr_ptr);
    uint64_t snax_symtab_start = (uint64_t)readw(local_symtab_start_addr_ptr);
    uint64_t snax_symtab_end   = (uint64_t)readw(local_symtab_end_addr_ptr);
    uint64_t snax_symtab_start_local = chiplet_addr_transform(snax_symtab_start);
    uint64_t snax_symtab_end_local   = chiplet_addr_transform(snax_symtab_end);
    // printf("Chip(%x, %x): [Host] Device symbol table range: 0x%lx - 0x%lx\r\n", get_current_chip_loc_x(), get_current_chip_loc_y(), snax_symtab_start, snax_symtab_end);
    snax_symbol_t *symtab_start = (snax_symbol_t *)(uintptr_t)snax_symtab_start_local;
    snax_symbol_t *symtab_end   = (snax_symbol_t *)(uintptr_t)snax_symtab_end_local;
    // printf("Scanning device symbol table...\n");
    for (volatile snax_symbol_t *sym = symtab_start; sym < symtab_end; sym++) {
        // printf("Symbol raw name: %s\n", (const char *)sym->name);
        // printf("Symbol addr     : 0x%x\n", (uint32_t)sym->addr);
        if (strcmp((const char *)sym->name, SNAX_SYMTAB_END_FN_NAME) == 0 &&
            sym->addr == SNAX_SYMTAB_END_FN_ADDR) {
            break;
        }

        // printf("Checking symbol: %s at 0x%x\n", sym->name, (unsigned)(uintptr_t)sym->addr);

        if (strcmp((const char *)sym->name, name) == 0) {
            return sym->addr;
        }
    }
    // printf("Symbol \"%s\" not found.\n", name);
    return (uint32_t)(0xBAADF00D);
}

//===============================================================
// RISC-V
//===============================================================

#define MIP_MTIP_OFFSET 7
#define MIP_MSIP_OFFSET 3
#define MIE_MSIE_OFFSET 3
#define MIE_MTIE_OFFSET 7
#define MSTATUS_MIE_OFFSET 3
#define MSTATUS_FS_OFFSET 13

// //===============================================================
// // Memory map pointers
// //===============================================================

// #if SELECT_FLL == 0  // ETH FLL
// volatile uint32_t* const fll_system_base =
//     (volatile uint32_t*)FLL_SYSTEM_BASE_ADDR;
// volatile uint32_t* const fll_periph_base =
//     (volatile uint32_t*)FLL_PERIPH_BASE_ADDR;
// volatile uint32_t* const fll_hbm2e_base =
//     (volatile uint32_t*)FLL_HBM2E_BASE_ADDR;
// #elif SELECT_FLL == 1  // GF FLL
// volatile uint32_t* const fll_system_base =
//     (volatile uint32_t*)FLL_SYSTEM_BASE_ADDR + (0x200 >> 2);
// volatile uint32_t* const fll_periph_base =
//     (volatile uint32_t*)FLL_PERIPH_BASE_ADDR + (0x200 >> 2);
// volatile uint32_t* const fll_hbm2e_base =
//     (volatile uint32_t*)FLL_HBM2E_BASE_ADDR + (0x200 >> 2);
// #endif

// volatile uint32_t* const fll_base[N_CLOCKS] = {fll_system_base,
// fll_periph_base, fll_hbm2e_base};

volatile uint64_t* const clint_mtime_ptr =
    (volatile uint64_t*)(CLINT_BASE_ADDR + CLINT_MTIME_LOW_REG_OFFSET);
volatile uint64_t* const clint_mtimecmp0_ptr =
    (volatile uint64_t*)(CLINT_BASE_ADDR + CLINT_MTIMECMP_LOW0_REG_OFFSET);

//===============================================================
// Globals
//===============================================================

// volatile comm_buffer_t comm_buffer __attribute__((aligned(8)));

//===============================================================
// Anticipated function declarations
//===============================================================

static inline void set_sw_interrupts_unsafe(uint8_t chip_id,
                                            uint32_t base_hartid,
                                            uint32_t num_harts,
                                            uint32_t stride);

static inline void set_sw_interrupt(uint8_t chip_id,
                                    volatile comm_buffer_t* comm_buffer_ptr,
                                    uint32_t hartid);

void delay_ns(uint64_t delay);

static inline volatile uint32_t* get_shared_lock(
    volatile comm_buffer_t* comm_buffer_ptr);

static inline void wait_sw_interrupt();

static inline void clear_sw_interrupt(uint8_t chip_id,
                                      volatile comm_buffer_t* comm_buffer_ptr,
                                      uint32_t hartid);

//===============================================================
// Initialization
//===============================================================

void initialize_bss() {
    extern volatile uint64_t __bss_start, __bss_end;

    size_t bss_size = (size_t)(&__bss_end) - (size_t)(&__bss_start);
    if (bss_size)
        sys_dma_blk_memcpy(
            get_current_chip_id(),
            chiplet_addr_transform((uint64_t)(&__bss_start)),
                           chiplet_addr_transform((uint64_t)WIDE_ZERO_MEM_BASE_ADDR),
                           bss_size);

}

void initialize_wide_spm() {
    size_t wide_spm_size =
        (size_t)(&__wide_spm_end) - (size_t)(&__wide_spm_start);
    if (wide_spm_size)
        sys_dma_blk_memcpy(
            get_current_chip_id(),
            chiplet_addr_transform((uint64_t)SPM_WIDE_BASE_ADDR),
            chiplet_addr_transform((uint64_t)WIDE_ZERO_MEM_BASE_ADDR),
            wide_spm_size);
}

void initialize_narrow_spm() {
    size_t narrow_spm_size =
        (size_t)(&__narrow_spm_end) - (size_t)(&__narrow_spm_start);
    if (narrow_spm_size)
        sys_dma_blk_memcpy(
            get_current_chip_id(),
            chiplet_addr_transform((uint64_t)SPM_NARROW_BASE_ADDR),
            chiplet_addr_transform((uint64_t)WIDE_ZERO_MEM_BASE_ADDR),
            narrow_spm_size);
    asm volatile("fence" ::: "memory");
}

void initialize_comm_buffer(comm_buffer_t* comm_buffer_ptr) {
    sys_dma_blk_memcpy(
        get_current_chip_id(),
        (uint64_t)comm_buffer_ptr,
        chiplet_addr_transform((uint64_t)WIDE_ZERO_MEM_BASE_ADDR),
        sizeof(comm_buffer_t));
    asm volatile("fence" ::: "memory");
}

void initialize_cluster(uint32_t cluster_idx) {
    // Initialize the cluster tcdm
    sys_dma_blk_memcpy(
        get_current_chip_id(),
        chiplet_addr_transform((uint64_t)cluster_tcdm_start_addr(cluster_idx)),
        chiplet_addr_transform((uint64_t)WIDE_ZERO_MEM_BASE_ADDR),
        (uint64_t)CLUSTER_TCDM_SIZE
    );
}

void initialize_all_clusters() {
    for (uint32_t i = 0; i < N_CLUSTERS; i++) {
        initialize_cluster(i);
    }
}



void enable_fpu() {
    uint64_t mstatus;

    asm volatile("csrr %[mstatus], mstatus" : [mstatus] "=r"(mstatus));
    mstatus |= (1 << MSTATUS_FS_OFFSET);
    asm volatile("csrw mstatus, %[mstatus]" : : [mstatus] "r"(mstatus));
}

void set_d_cache_enable(uint16_t ena) {
    asm volatile("csrw 0x701, %0" ::"r"(ena));
}

//===============================================================
// Synchronization and mutual exclusion
//===============================================================

static inline void fence() { asm volatile("fence" : : : "memory"); }

/**
 * @brief lock a mutex, blocking
 * @details test-and-set (tas) implementation of a lock.
 *          Declare mutex with `static volatile uint32_t mtx = 0;`
 */
void mutex_tas_acquire(volatile uint32_t* pmtx) {
    asm volatile(
        "li            x5,1          # x5 = 1\n"
        "1:\n"
        "  amoswap.w.aq  x5,x5,(%0)   # x5 = oldlock & lock = 1\n"
        "  bnez          x5,1b      # Retry if previously set)\n"
        : "+r"(pmtx)
        :
        : "x5");
}

/**
 * @brief lock a mutex, blocking
 * @details test-and-test-and-set (ttas) implementation of a lock.
 *          Declare mutex with `static volatile uint32_t mtx = 0;`
 */
static inline void mutex_ttas_acquire(volatile uint32_t* pmtx) {
    asm volatile(
        "1:\n"
        "  lw x5, 0(%0)\n"
        "  bnez x5, 1b\n"
        "  li x5,1          # x5 = 1\n"
        "2:\n"
        "  amoswap.w.aq  x5,x5,(%0)   # x5 = oldlock & lock = 1\n"
        "  bnez          x5,2b      # Retry if previously set)\n"
        : "+r"(pmtx)
        :
        : "x5");
}

/**
 * @brief Release the mutex
 */
static inline void mutex_release(volatile uint32_t* pmtx) {
    asm volatile("amoswap.w.rl  x0,x0,(%0)   # Release lock by storing 0\n"
                 : "+r"(pmtx));
}

//===============================================================
// Device programming
//===============================================================

extern void snitch_main();

static inline void wakeup_snitch(uint8_t chip_id,
                                 volatile comm_buffer_t* comm_buffer_ptr,
                                 uint32_t hartid) {
    set_sw_interrupt(chip_id, comm_buffer_ptr, hartid);
}

/**
 * @brief Waits until snitches are parked in a `wfi` instruction
 *
 * @detail delays execution to wait for the Snitch cores to be ready.
 *         After being parked, the Snitch cores can accept an interrupt
 *         and start executing its binary
 */
// TODO: implement in a more robust manner
void wait_snitches_parked(uint32_t timeout) { delay_ns(100000); }

/**
 * @brief Programs the Snitches with the Snitch binary
 *
 * @detail After boot, the Snitches are "parked" on a WFI
 *         until they receive a software interrupt. Upon
 *         wakeup, the Snitch jumps to a minimal interrupt
 *         handler in boot ROM which loads the address of the
 *         user binary from the soc_ctrl_scratch_0 register.
 *         This routine programs the soc_ctrl_scratch_0 register
 *         with the address of the user binary.
 */
static inline void program_snitches(uint8_t chip_id,
                                    volatile comm_buffer_t* comm_buffer_ptr) {
    writew((uint32_t)(uintptr_t)snitch_main,
           (uintptr_t)chiplet_addr_transform_full(chip_id, (uintptr_t)soc_ctrl_scratch_addr(1)));
    writew((uint32_t)(uintptr_t)comm_buffer_ptr,
           (uintptr_t)chiplet_addr_transform_full(chip_id, (uintptr_t)soc_ctrl_scratch_addr(2)));
}

/**
 * @brief Wake-up a Snitch cluster
 *
 * @detail Send a cluster interrupt to all Snitches in a cluster
 */

static inline void wakeup_cluster(uint8_t chip_id, uint32_t cluster_id) {
    writew(511, 
           (uintptr_t)chiplet_addr_transform_full(chip_id, (uintptr_t)cluster_clint_set_addr(cluster_id)));
}

/**
 * @brief Wake-up Snitches
 *
 * @detail All Snitches are "parked" in a WFI. A SW interrupt
 *         must be issued to "unpark" every Snitch. This function
 *         sends a SW interrupt to all Snitches.
 */
void wakeup_snitches(uint8_t chip_id, volatile comm_buffer_t* comm_buffer_ptr) {
    volatile uint32_t* lock = get_shared_lock(comm_buffer_ptr);

    mutex_ttas_acquire(lock);
    set_sw_interrupts_unsafe(chip_id, 1, N_SNITCHES, 1);
    mutex_release(lock);
}

/**
 * @brief Wake-up Snitches
 *
 * @detail Send a cluster interrupt to all Snitches
 */
static inline void wakeup_snitches_cl(uint8_t chip_id) {
    for (int i = 0; i < N_CLUSTERS; i++) wakeup_cluster(chip_id, i);
}

/**
 * @brief Wake-up Snitches
 *
 * @detail All Snitches are "parked" in a WFI. A SW interrupt
 *         must be issued to "unpark" every Snitch. This function
 *         sends a SW interrupt to a given range of Snitches.
 */
void wakeup_snitches_selective(uint8_t chip_id,
                               volatile comm_buffer_t* comm_buffer_ptr,
                               uint32_t base_hartid, uint32_t num_harts,
                               uint32_t stride) {
    volatile uint32_t* lock = get_shared_lock(comm_buffer_ptr);

    mutex_ttas_acquire(lock);
    set_sw_interrupts_unsafe(chip_id, base_hartid, num_harts, stride);
    mutex_release(lock);
}

// temporary deprecate this function since it uses the N_CORES_PER_CLUSTER

// /**
//  * @brief Wake-up Snitches
//  *
//  * @detail All Snitches are "parked" in a WFI. A SW interrupt
//  *         must be issued to "unpark" every Snitch. This function
//  *         sends a SW interrupt to one Snitch in every cluster,
//  *         the so called "master" of the cluster. The "master" is
//  *         then expected to wake-up all the other Snitches in its
//  *         cluster. The "master" Snitches can use the cluster-local
//  *         CLINTs without sending requests outside the cluster,
//  *         avoiding congestion.
//  */
// void wakeup_master_snitches() {
//     volatile uint32_t* lock = get_shared_lock();

//     mutex_ttas_acquire(lock);
//     set_sw_interrupts_unsafe(1, N_CLUSTERS, N_CORES_PER_CLUSTER);
//     mutex_release(lock);
// }

/**
 * @brief Waits until snitches are done executing
 */
static inline int wait_snitches_done(uint8_t chip_id) {
    wait_sw_interrupt();
    clear_host_sw_interrupt(get_current_chip_id());

    uint32_t retval = readw(
        (uintptr_t)chiplet_addr_transform_full(chip_id, (uintptr_t)soc_ctrl_scratch_addr(3)));
    // LSB signals completion
    if (retval & 1)
        return retval >> 1;
    else
        return -1;
}

static inline volatile uint32_t* get_shared_lock(
    volatile comm_buffer_t* comm_buffer_ptr) {
    return &((*comm_buffer_ptr).lock);
}

//===============================================================
// Interrupts
//===============================================================

static inline void wfi() { asm volatile("wfi"); }

static inline void enable_sw_interrupts() {
    uint64_t mie;

    asm volatile("csrr %[mie], mie" : [mie] "=r"(mie));
    mie |= (1 << MIE_MSIE_OFFSET);
    asm volatile("csrw mie, %[mie]" : : [mie] "r"(mie));
}

static inline uint32_t get_clint_msip_hart(uint8_t chip_id, uint32_t hartid) {
    uintptr_t base_addr = (uintptr_t)get_chip_baseaddress(chip_id);
    uint32_t field_offset = hartid % CLINT_MSIP_P_FIELDS_PER_REG;
    uint32_t lsb_offset = field_offset * CLINT_MSIP_P_FIELD_WIDTH;
    return (*(volatile uint32_t*)((uintptr_t)clint_msip_ptr(hartid) |
                                  base_addr) >>
            lsb_offset) &
           1;
}

/**
 * @brief Gets SW interrupt pending status from local CSR
 *
 * @detail Use this in favour of remote_sw_interrupt_pending()
 *         when polling a core's own interrupt pending
 *         status. This avoids unnecessary congestion on the
 *         interconnect and shared CLINT.
 */
static inline uint32_t sw_interrupt_pending() {
    uint64_t mip;

    asm volatile("csrr %[mip], mip" : [mip] "=r"(mip));
    return mip & (1 << MIP_MSIP_OFFSET);
}

// TODO: for portability to architectures where WFI is implemented as a NOP
//       also sw_interrupts_enabled() should be checked
static inline void wait_sw_interrupt() {
    do wfi();
    while (!sw_interrupt_pending());
}

static inline void clear_sw_interrupt_unsafe(uint8_t chip_id, uint32_t hartid) {
    uintptr_t base_addr = (uintptr_t)get_chip_baseaddress(chip_id);
    uint32_t field_offset = hartid % CLINT_MSIP_P_FIELDS_PER_REG;
    uint32_t lsb_offset = field_offset * CLINT_MSIP_P_FIELD_WIDTH;

    *(volatile uint32_t*)((uintptr_t)clint_msip_ptr(hartid) | base_addr) &=
        ~(1 << lsb_offset);
}

static inline void clear_sw_interrupt(uint8_t chip_id,
                                      volatile comm_buffer_t* comm_buffer_ptr,
                                      uint32_t hartid) {
    volatile uint32_t* shared_lock = get_shared_lock(comm_buffer_ptr);

    mutex_tas_acquire(shared_lock);
    clear_sw_interrupt_unsafe(chip_id, hartid);
    mutex_release(shared_lock);
}

/**
 * @brief Gets SW interrupt pending status from CLINT
 *
 * @detail Use sw_interrupt_pending() in favour of this
 *         when polling a core's own interrupt pending
 *         status. That function interrogates a local CSR
 *         instead of the shared CLINT.
 */
static inline uint32_t remote_sw_interrupt_pending(uint8_t chip_id,
                                                   uint32_t hartid) {
    return get_clint_msip_hart(chip_id, hartid);
}

static inline uint32_t timer_interrupts_enabled() {
    uint64_t mie;
    asm volatile("csrr %[mie], mie" : [mie] "=r"(mie));
    return (mie >> MIE_MTIE_OFFSET) & 1;
}

static inline void set_sw_interrupt_unsafe(uint8_t chip_id, uint32_t hartid) {
    uintptr_t base_addr = (uintptr_t)get_chip_baseaddress(chip_id);
    uint32_t field_offset = hartid % CLINT_MSIP_P_FIELDS_PER_REG;
    uint32_t lsb_offset = field_offset * CLINT_MSIP_P_FIELD_WIDTH;

    *(volatile uint32_t*)((uintptr_t)clint_msip_ptr(hartid) | base_addr) |=
        (1 << lsb_offset);
}

void set_sw_interrupt(uint8_t chip_id, volatile comm_buffer_t* comm_buffer_ptr,
                      uint32_t hartid) {
    volatile uint32_t* shared_lock = get_shared_lock(comm_buffer_ptr);
    mutex_ttas_acquire(shared_lock);
    set_sw_interrupt_unsafe(chip_id, hartid);
    mutex_release(shared_lock);
}

static inline void set_sw_interrupts_unsafe(uint8_t chip_id,
                                            uint32_t base_hartid,
                                            uint32_t num_harts,
                                            uint32_t stride) {
    uintptr_t base_addr = (uintptr_t)get_chip_baseaddress(chip_id);
    volatile uint32_t* ptr =
        (uint32_t*)((uintptr_t)clint_msip_ptr(base_hartid) | base_addr);

    uint32_t num_fields = num_harts;
    uint32_t field_idx = base_hartid;
    uint32_t field_offset = field_idx % CLINT_MSIP_P_FIELDS_PER_REG;
    uint32_t reg_idx = field_idx / CLINT_MSIP_P_FIELDS_PER_REG;
    uint32_t prev_reg_idx = reg_idx;
    uint32_t mask = 0;
    uint32_t reg_jump;
    uint32_t last_field = num_fields - 1;

    for (uint32_t i = 0; i < num_fields; i++) {
        // put field in mask
        mask |= 1 << field_offset;

        // calculate next field info
        field_idx += stride;
        field_offset = field_idx % CLINT_MSIP_P_FIELDS_PER_REG;
        reg_idx = field_idx / CLINT_MSIP_P_FIELDS_PER_REG;
        reg_jump = reg_idx - prev_reg_idx;

        // if next value is in another register
        if (i != last_field && reg_jump) {
            // store mask
            if (mask == (uint32_t)(-1))
                *ptr = mask;
            else
                *ptr |= mask;
            // update pointer and reset mask
            ptr += reg_jump;
            prev_reg_idx = reg_idx;
            mask = 0;
        }
    }

    // store last mask
    *ptr |= mask;
}

void set_cluster_interrupt(uint8_t chip_id, uint32_t cluster_id,
                           uint32_t core_id) {
    writew((1 << core_id),
           (uintptr_t)chiplet_addr_transform_full(chip_id, (uint64_t)cluster_clint_set_addr(cluster_id)));
}

static inline uint32_t timer_interrupt_pending() {
    uint64_t mip;

    asm volatile("csrr %[mip], mip" : [mip] "=r"(mip));
    return mip & (1 << MIP_MTIP_OFFSET);
}

void wait_timer_interrupt() {
    do wfi();
    while (!timer_interrupt_pending() && timer_interrupts_enabled());
}

void enable_global_interrupts() {
    uint64_t mstatus;

    asm volatile("csrr %[mstatus], mstatus" : [mstatus] "=r"(mstatus));
    mstatus |= (1 << MSTATUS_MIE_OFFSET);
    asm volatile("csrw mstatus, %[mstatus]" : : [mstatus] "r"(mstatus));
}

void enable_timer_interrupts() {
    uint64_t mie;

    asm volatile("csrr %[mie], mie" : [mie] "=r"(mie));
    mie |= (1 << MIE_MTIE_OFFSET);
    asm volatile("csrw mie, %[mie]" : : [mie] "r"(mie));
}

void disable_timer_interrupts() {
    uint64_t mie;

    asm volatile("csrr %[mie], mie" : [mie] "=r"(mie));
    mie &= ~(1 << MIE_MTIE_OFFSET);
    asm volatile("csrw mie, %[mie]" : : [mie] "r"(mie));
}

void disable_sw_interrupts() {
    uint64_t mie;

    asm volatile("csrr %[mie], mie" : [mie] "=r"(mie));
    mie &= ~(1 << MIE_MSIE_OFFSET);
    asm volatile("csrw mie, %[mie]" : : [mie] "r"(mie));
}

/**
 * @brief Gets SW interrupt pending status from local CSR
 *
 * @detail Use this in favour of wait_remote_sw_interrupt_pending()
 *         when polling a core's own interrupt pending
 *         status. This avoids unnecessary congestion on the
 *         interconnect and shared CLINT.
 */
void wait_sw_interrupt_cleared() { while (sw_interrupt_pending()); }

/**
 * @brief Gets SW interrupt pending status from shared CLINT
 *
 * @detail Use wait_sw_interrupt_cleared() in favour of this
 *         when polling a core's own interrupt pending
 *         status. That function polls a local CSR instead
 *         of the shared CLINT.
 */
void wait_remote_sw_interrupt_pending(uint8_t chip_id, uint32_t hartid) {
    while (remote_sw_interrupt_pending(chip_id, hartid));
}

//===============================================================
// Timers
//===============================================================

static const float rtc_period = 30517.58;  // ns

static inline uint64_t mcycle() {
    register uint64_t r;
    asm volatile("csrr %0, mcycle" : "=r"(r));
    return r;
}

static inline uint64_t mtime(uint8_t chip_id) {
    uintptr_t base_addr = (uintptr_t)get_chip_baseaddress(chip_id);
    return *(volatile uint64_t*)((uintptr_t)clint_mtime_ptr | base_addr);
}

void set_timer_interrupt(uint8_t chip_id, uint64_t interval_ns) {
    // Convert ns to RTC unit
    uint64_t rtc_interval = interval_ns / (int64_t)rtc_period;

    // Calculate the base address for the chip
    uintptr_t base_addr = (uintptr_t)get_chip_baseaddress(chip_id);

    // Offset interval by current time and set the timer interrupt
    *(volatile uint64_t*)((uintptr_t)clint_mtimecmp0_ptr | base_addr) =
        mtime(chip_id) + rtc_interval;
}

/**
 * @brief Clears timer interrupt
 *
 * @detail Pending timer interrupts are cleared in HW when
 *         writing to the mtimecmp register. Note that
 *         eventually the mtime register is going to be greater
 *         than the newly programmed mtimecmp register, reasserting
 *         the pending bit. If this is not desired, it is safer
 *         to disable the timer interrupt before clearing it.
 */
void clear_timer_interrupt(uint8_t chip_id) {
    uintptr_t base_addr = (uintptr_t)get_chip_baseaddress(chip_id);
    *(volatile uint64_t*)((uintptr_t)clint_mtimecmp0_ptr | base_addr) =
        mtime(chip_id) + 1;
}

// Minimum delay is of one RTC period
void delay_ns(uint64_t delay) {
    uint8_t chip_id = get_current_chip_id();
    set_timer_interrupt(chip_id, delay);

    // Wait for set_timer_interrupt() to have effect
    fence();
    enable_timer_interrupts();

    wait_timer_interrupt();
    disable_timer_interrupts();
    clear_timer_interrupt(chip_id);
}

//===============================================================
// Chip Level Synchronization Mechanism
//===============================================================

void announce_chip_checkpoint(volatile comm_buffer_t* chip_barrier_data_ptr,
                              uint8_t checkpoint) {
    volatile uint8_t* this_chip_checkpoint =
        &((*chip_barrier_data_ptr).chip_level_checkpoint[get_current_chip_id()]);
    // Broadcast to all Chips
    this_chip_checkpoint =
        (uint8_t*)(((uint64_t)this_chip_checkpoint) | (((uint64_t)0xFF) << 40));
    *this_chip_checkpoint = checkpoint;
}

void wait_chip_checkpoint(volatile comm_buffer_t* chip_barrier_data_ptr,
                          uint8_t chip_id, uint8_t checkpoint) {
    volatile uint8_t* target_chip_checkpoint =
        &((*chip_barrier_data_ptr).chip_level_checkpoint[chip_id]);
    // Broadcast to all Chips
    while (*target_chip_checkpoint != checkpoint) {
        asm volatile("fence.i" ::: "memory");
    }
}

void wait_chips_checkpoint(volatile comm_buffer_t* chip_barrier_data_ptr,
                           uint8_t top_left_chip_id,
                           uint8_t bottom_right_chip_id, uint8_t checkpoint) {
    volatile uint8_t* chip_level_checkpoint =
        &((*chip_barrier_data_ptr).chip_level_checkpoint[0]);
    uint8_t current_chip_id = get_current_chip_id();
    uint8_t continue_loop = 1;
    while (continue_loop) {
        continue_loop = 0;
        asm volatile("fence.i" ::: "memory");
        for (uint8_t i = top_left_chip_id >> 4;
             i <= (bottom_right_chip_id >> 4); i++) {
            for (uint8_t j = top_left_chip_id & 0xF;
                 j <= (bottom_right_chip_id & 0xF); j++) {
                if ((*(chip_level_checkpoint + ((i << 4) + j)) != checkpoint) &&
                    (current_chip_id != ((i << 4) + j))) {
                    continue_loop = 1;
                    break;
                }
            }
        }
    }
}

// Barrier is realized in software, to ensure that all other chips have reached
// a certain checkpoint
void chip_barrier(volatile comm_buffer_t* chip_barrier_data_ptr,
                  uint8_t top_left_chip_id, uint8_t bottom_right_chip_id,
                  uint8_t checkpoint) {
    volatile uint8_t* chip_level_checkpoint =
        &((*chip_barrier_data_ptr).chip_level_checkpoint[0]);
    // Broadcast to all other chip on the progress of the chip
    announce_chip_checkpoint(chip_barrier_data_ptr, checkpoint);
    // Change the pointer back
    wait_chips_checkpoint(chip_barrier_data_ptr, top_left_chip_id,
                          bottom_right_chip_id, checkpoint);
}
