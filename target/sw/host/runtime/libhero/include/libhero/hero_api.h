// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Pirmin Vogel <vogelpi@iis.ee.ethz.ch>
// Cyril Koenig <cykoenig@iis.ee.ethz.ch>

#pragma once
#include <stdbool.h> // for bool
#include <stdint.h>
#include <stddef.h>
#include "libhero/ringbuf.h"
#include "libhero/debug.h"
#include "o1heap.h"
#include "snitch_cluster.h"
// ////////////////////
// ///// OMP API //////
// ////////////////////

// #ifndef HERO_DEV_DEFAULT_FREQ_MHZ
// #define HERO_DEV_DEFAULT_FREQ_MHZ 50
// #endif

// #define HERODEV_SVM     (0)
// #define HERODEV_MEMCPY  (1)
// #define HOST            (2)

// typedef enum {
//     inout = 0,
//     in = 1,
//     out = 2,
// } ElemType;

// // striping informationg structure
// typedef struct {
//     unsigned n_stripes;
//     unsigned first_stripe_size_b;
//     unsigned last_stripe_size_b;
//     unsigned stripe_size_b;
// } StripingDesc;

// typedef enum {
//     copy = 0x0,       // no SVM, copy-based sharing using contiguous L3 memory
//     svm_static = 0x1, // SVM, set up static mapping at offload time, might fail
//                       // - use with caution
//     svm_stripe =
//         0x2,      // SVM, use striping (L1 only), might fail - use with caution
//     svm_mh = 0x3, // SVM, use miss handling
//     copy_tryx = 0x4, // no SVM, copy-based sharing using contiguous L3 memory,
//                      // but let PULP do the tryx()
//     svm_smmu = 0x5,  // SVM, use SMMU instead of RAB
//     svm_smmu_shpt = 0x6, // SVM, use SMMU, emulate sharing of user-space page
//                          // table, no page faults
//     custom = 0xF,        // do not touch (custom marshalling)
// } ShMemType;

// // shared variable data structure
// typedef struct {
//     void *ptr;      // address in host virtual memory
//     void *ptr_l3_v; // host virtual address in contiguous L3 memory   - filled
//                     // by runtime library
//                     // based on sh_mem_ctrl
//     void *ptr_l3_p; // PULP physical address in contiguous L3 memory  - filled
//                     // by runtime library
//                     // based on sh_mem_ctrl
//     unsigned size;  // size in Bytes
//     ElemType type;  // inout, in, out
//     ShMemType sh_mem_ctrl;
//     unsigned char cache_ctrl;  // 0: flush caches, access through DRAM
//                                // 1: do not flush caches, access through caches
//     unsigned char rab_lvl;     // 0: first L1, L2 when full
//                                // 1: L1 only
//                                // 2: L2 only
//     StripingDesc *stripe_desc; // only used if sh_mem_ctrl = 2
// } DataDesc;


////////////////////
///// MBOXES  //////
////////////////////

#define MBOX_DEVICE_READY (0x01U)
#define MBOX_DEVICE_START (0x02U)
#define MBOX_DEVICE_BUSY (0x03U)
#define MBOX_DEVICE_DONE (0x04U)
#define MBOX_DEVICE_PRINT (0x05U)
#define MBOX_DEVICE_STOP (0x0FU)
#define MBOX_DEVICE_LOGLVL (0x10U)
#define MBOX_HOST_READY (0x1000U)
#define MBOX_HOST_DONE (0x3000U)
#define HOST_MBOX_CYCLES 20
////////////////////
///// DEVICES  /////
////////////////////

typedef struct HeroSubDev {
    unsigned *v_addr;
    size_t p_addr;
    unsigned size;
    char *alias;
    struct HeroSubDev *next;
} HeroSubDev_t;

typedef struct {
    volatile struct ring_buf *a2h_mbox;
    HeroSubDev_t a2h_mbox_mem;
    volatile struct ring_buf *h2a_mbox;
    HeroSubDev_t h2a_mbox_mem;
    volatile struct ring_buf *rb_mbox;
    HeroSubDev_t rb_mbox_mem;
} HeroMboxes_t;

typedef struct {
    void *dev;
    uint8_t dev_id;
    HeroSubDev_t *global_mems;
    char alias[32];
    HeroMboxes_t mboxes;
} HeroDev;

////////////////////
///// API      /////
////////////////////
int hero_dev_alloc_mboxes(HeroDev *dev);
int hero_dev_free_mboxes(HeroDev *dev);
/** @name Mailbox communication functions
 *
 * @{
 */

/** Read one or multiple words from the mailbox. Blocks if the mailbox does not
 contain enough
 *  data.

  \param    pulp    pointer to the HeroDev structure
  \param    buffer  pointer to read buffer
  \param    n_words number of words to read

  \return   0 on success; negative value with an errno on errors.
 */
int hero_dev_mbox_read(const HeroDev *dev, uint32_t *buffer, size_t n_words);



/** Non-blocking read one words from the mailbox

  \param    pulp    pointer to the HeroDev structure
  \param    buffer  pointer to read buffer
  \param    n_words number of words to read

  \return   0 on success; negative value with an errno on errors.
 */
int hero_dev_mbox_try_read(const HeroDev *dev, uint32_t *buffer);



/** Write one word to the mailbox. Blocks if the mailbox is full.

 \param     pulp pointer to the HeroDev structure
 \param     word word to write

 \return    0 on success; negative value with an errno on errors.
 */
int hero_dev_mbox_write(HeroDev *dev, uint32_t word);

//!@}

/** @name PULP library setup functions
 *
 * @{
 */

/** Init the L2 and L3 Heap Allocator.

  \return   0 on success; negative value with an errno on errors.
 */
int hero_dev_mmap_init();

/** Memory map the PULP device to virtual user space using the mmap() syscall.

  \param    pulp pointer to the HeroDev structure

  \return   0 on success; negative value with an errno on errors.
 */
int hero_dev_mmap(HeroDev *dev);

/** Undo the memory mapping of the PULP device to virtual user space using the
  munmap() syscall.

  \param    pulp pointer to the HeroDev structure

  \return   0 on success; negative value with an errno on errors.
 */
int hero_dev_munmap(HeroDev *dev);

//!@}

/** @name PULP hardware setup functions
 *
 * @{
 */

/** Set the clock frequency of PULP. Blocks until the PLL inside PULP locks.

 * NOTE: Only do this at startup of PULP!

  \param    pulp         pointer to the HeroDev structure
  \param    des_freq_mhz desired clock frequency in MHz

  \return   configured frequency in MHz.
 */
int hero_dev_clking_set_freq(HeroDev *dev, uint32_t des_freq_mhz);

/** Measure the clock frequency of PULP. Can only be executed with the RAB
 configured to allow
 *  accessing the cluster peripherals. To validate the measurement, the ZYNQ_PMM
 needs to be loaded
 *  for accessing to the ARM clock counters.

  \param    pulp pointer to the HeroDev structure

  \return   measured clock frequency in MHz.
 */
int hero_dev_clking_measure_freq(HeroDev *dev);

/** Initialize the memory mapped PULP device: disable reset and fetch enable,
 set up basic RAB
 *  mappings, enable mailbox interrupts, disable RAB miss interrupts, pass
 information to driver.

 *  NOTE: PULP needs to be mapped to memory before this function can be
 executed.

  \param    pulp pointer to the HeroDev structure

  \return   0 on success; negative value with an errno on errors.
 */
int hero_dev_init(HeroDev *dev);
int hero_dev_l2_init();
int hero_dev_l3_init();

/** Reset PULP.

  \param    pulp pointer to the HeroDev structure
  \param    full type of reset: 0 for PULP reset, 1 for entire FPGA
 */
void hero_dev_reset(HeroDev *dev, bool full);

//!@}

/** Load binaries to the start of the TCDM and the start of the L2 memory inside
 PULP. This
 *  function uses an mmap() system call to map the specified binaries to memory.

  \param    pulp pointer to the HeroDev structure
  \param    name pointer to the string containing the name of the application to
 load

  \return   0 on success; negative value with an errno on errors.
 */
int hero_dev_load_bin(HeroDev *dev, const char *name);

/** Load binary to the start of the L2 memory inside PULP. This functions
 expects the binary to be
 *  loaded in host memory.

 \param    pulp pointer to the HeroDev structure
 \param    ptr  pointer to mem where the binary is loaded
 \param    size binary size in bytes

 \return   0 on success.
 */
int hero_dev_load_bin_from_mem(HeroDev *dev, void *ptr, unsigned size);

/** Starts programm execution on PULP.

 \param    pulp pointer to the HeroDev structure
 */
void hero_dev_exe_start(HeroDev *dev);

/** Stops programm execution on PULP.

 \param    pulp pointer to the HeroDev structure
 */
void hero_dev_exe_stop(HeroDev *dev);

/** Polls the GPIO register for the end of computation signal for at most
 timeout_s seconds.

 \param    pulp      pointer to the HeroDev structure
 \param    timeout_s maximum number of seconds to wait for end of computation

 \return   0 on success; -ETIME in case of an execution timeout or -ENOSYS if
 EOC is not available.
 */
int hero_dev_exe_wait(const HeroDev *dev, uint32_t timeout_s);

/** @name Contiguous memory allocation functions
 *
 * @{
 */

/** Allocate a chunk of memory in contiguous L3.
  \param    pulp   pointer to the HeroDev structure
  \param    size_b size in Bytes of the requested chunk
  \param    p_addr pointer to store the physical address to
  \return   virtual user-space address for host
 */
uintptr_t hero_dev_l3_malloc(HeroDev *dev, uint32_t size_b, uintptr_t *p_addr);

uintptr_t hero_dev_l2_malloc(HeroDev *dev, uint32_t size_b, uintptr_t *p_addr);

uintptr_t hero_dev_l2_mailbox_malloc(HeroDev *dev, uint32_t size_b, uintptr_t *p_addr);

/** Allocate a DMA-able buffer host L3.
  \param    pulp   pointer to the HeroDev structure
  \param    size_b size in Bytes of the requested chunk
  \param    p_addr pointer to store the physical address to
  \return   virtual user-space address for host
 */
uintptr_t hero_host_l3_malloc(HeroDev *dev, uint32_t size_b, uintptr_t *p_addr);

uintptr_t hero_iommu_map_virt(HeroDev *dev, uint32_t size_b, void *v_addr);

int hero_iommu_map_virt_to_phys(HeroDev *dev, uint32_t size_b, void *v_addr, uintptr_t p_addr);

/** Free memory previously allocated in contiguous L3.
 \param    pulp   pointer to the HeroDev structure
 \param    v_addr pointer to unsigned containing the virtual address
 \param    p_addr pointer to unsigned containing the physical address
 */
void hero_dev_l3_free(HeroDev *dev, uintptr_t v_addr, uintptr_t p_addr);

//!@}

/** @name Host DMA functions
 *
 * @{
 */

/** Setup a DMA transfer using the Host DMA engine

 \param    pulp      pointer to the HeroDev structure
 \param    addr_l3   virtual address in host's L3
 \param    addr_pulp physical address in PULP, so far, only L2 tested
 \param    size_b    size in bytes
 \param    host_read 0: Host -> PULP, 1: PULP -> Host (not tested)

 \return   0 on success; negative value with an errno on errors.
 */
int hero_dev_dma_xfer(const HeroDev *dev, uintptr_t addr_l3,
                      uintptr_t addr_pulp, size_t size_b, bool host_read);

//!@}