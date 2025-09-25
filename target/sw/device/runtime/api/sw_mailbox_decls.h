// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>

#define MBOX_DEVICE_READY (0x01U)
#define MBOX_DEVICE_START (0x02U)
#define MBOX_DEVICE_BUSY (0x03U)
#define MBOX_DEVICE_DONE (0x04U)
#define MBOX_DEVICE_STOP (0x0FU)
#define MBOX_DEVICE_LOGLVL (0x10U)
#define MBOX_HOST_READY (0x1000U)
#define MBOX_HOST_DONE (0x3000U)

#define SYS_exit 60
#define SYS_write 64
#define SYS_read 63
#define SYS_wake 1235
#define SYS_cycle 1236

/***********************************************************************************
 * TYPES
 ***********************************************************************************/

/**
 * @brief Ring buffer for simple communication from accelerator to host.
 * @tail: Points to the element in `data` which is read next
 * @head: Points to the element in `data` which is written next
 * @size: Number of elements in `data`. Head and tail pointer wrap at `size`
 * @element_size: Size of each element in bytes
 * @data_p: points to the base of the data buffer in physical address
 * @data_v: points to the base of the data buffer in virtual address space
 */
struct __attribute__((packed, aligned(4))) ring_buf {
  uint32_t head;
  uint32_t size;
  uint32_t tail;
  uint32_t element_size;
  uint32_t data_v;
  uint32_t data_p;
};

/**
 * @brief Holds physical addresses of the shared L2
 * @a2h_rb: acc to host ring buffer (for the system call)
 * @a2h_mbox: acc to host mailbox (for returning tasking info to host)
 * @h2a_mbox: host to acc mailbox (for host to dispatch task to acc)
 * @head: base of heap memory
 */
struct __attribute__((packed, aligned(4))) l2_layout {
    uint32_t a2h_rb;
    uint32_t a2h_mbox;
    uint32_t h2a_mbox;
    uint32_t heap;
};

/***********************************************************************************
 * PUBLICS
 ***********************************************************************************/
inline uint32_t syscall(uint32_t which, uint32_t arg0, uint32_t arg1, uint32_t arg2,
            uint32_t arg3, uint32_t arg4);
/**
 * @brief Blocking mailbox read access
 */
inline uint32_t mailbox_read(volatile struct ring_buf *g_h2a_mbox, uint32_t *buffer, uint32_t n_words);
/**
 * @brief Non-Blocking mailbox read access. Return 1 on success, 0 on fail
 */
inline uint32_t mailbox_try_read(volatile struct ring_buf *g_h2a_mbox, uint32_t *buffer);
/**
 * @brief Blocking mailbox write access
 */
inline uint32_t mailbox_write(volatile struct ring_buf *g_a2h_mbox, uint32_t word);

/**
 * @brief Blocking read a word from the HW H2C mailbox
 */
inline uint32_t bingo_h2c_mailbox_read(uint32_t *buffer);

/**
 * @brief Blocking write a word to the HW C2H mailbox
 */

inline void bingo_c2h_mailbox_write(uint32_t word);

