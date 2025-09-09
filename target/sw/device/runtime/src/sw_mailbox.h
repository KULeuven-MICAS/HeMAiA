// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>

inline void csleep(uint32_t cycles) {
    uint32_t start = snrt_mcycle();
    while ((snrt_mcycle() - start) < cycles) {}
}
inline void dump_mbox(struct ring_buf *rbuf) {
  printf("---DUMPING NOW---\n\r");
  printf("mbox (%x)\n\r", rbuf);
  uint8_t* addr = rbuf;
  for(int i = 0; i < sizeof(struct ring_buf); i++) {
    if(i % 8 == 0)
        printf("\n\r(%x) : ", addr);
    printf("%x-", *(addr++));
  }
  printf("\n\r");
  printf("head : %x = %u\n\r"     , &rbuf->head         , rbuf->head         );
  printf("size : %x = %u\n\r"     , &rbuf->size         , rbuf->size         );
  printf("tail : %x = %u\n\r"     , &rbuf->tail         , rbuf->tail         );
  printf("data_p : %x = %x\n\r", &rbuf->data_p , rbuf->data_p       );
  printf("data_v : %x = %x\n\r", &rbuf->data_v , rbuf->data_v       );
  //printf("tail %u, data_v %" PRIu64 ", element_size %u, size %u, data_p %" PRIu64 ", head %u\n\r", rbuf->tail, rbuf->data_v, rbuf->element_size, rbuf->size, rbuf->data_p, rbuf->head);
  printf("---DUMPING ENDS---\n\r");
}

/**
 * @brief Copy data from `el` in the next free slot in the ring-buffer on the
 *physical addresses
 *
 * @param rb pointer to the ring buffer struct
 * @param el pointer to the data to be copied into the ring buffer
 * @return int 0 on succes, -1 if the buffer is full
 */
inline uint32_t rb_device_put(volatile struct ring_buf *rb, void *el) {
    uint32_t next_head = (rb->head + 1) % rb->size;
    // caught the tail, can't put data
    if (next_head == rb->tail)
        return -1;
    for (uint32_t i = 0; i < rb->element_size; i++)
        *((uint8_t *)rb->data_p + rb->element_size *rb->head + i) =
            *((uint8_t *)el + i);
    rb->head = next_head;
    return 0;
}

/**
 * @brief Pop element from ring buffer on virtual addresses
 *
 * @param rb pointer to ring buffer struct
 * @param el pointer to where element is copied to
 * @return 0 on success, -1 if no element could be popped
 */
inline uint32_t rb_host_get(volatile struct ring_buf *rb, void *el) {
    // caught the head, can't get data
    if (rb->tail == rb->head)
        return -1;
    for (uint32_t i = 0; i < rb->element_size; i++)
        *((uint8_t *)el + i) =
            *((uint8_t *)rb->data_v + rb->element_size * rb->tail + i);
    rb->tail = (rb->tail + 1) % rb->size;
    return 0;
}

/**
 * @brief Copy data from `el` in the next free slot in the ring-buffer on the
 *virtual addresses
 *
 * @param rb pointer to the ring buffer struct
 * @param el pointer to the data to be copied into the ring buffer
 * @return int 0 on succes, -1 if the buffer is full
 */
inline int rb_host_put(volatile struct ring_buf *rb, void *el) {
    uint32_t next_head = (rb->head + 1) % rb->size;
    // caught the tail, can't put data
    if (next_head == rb->tail)
        return -1;
    for (uint32_t i = 0; i < rb->element_size; i++)
        *((uint8_t *)rb->data_v + rb->element_size *rb->head + i) =
            *((uint8_t *)el + i);
    rb->head = next_head;
    return 0;
}

/**
 * @brief Pop element from ring buffer on physicl addresses
 *
 * @param rb pointer to ring buffer struct
 * @param el pointer to where element is copied to
 * @return 0 on success, -1 if no element could be popped
 */
inline uint32_t rb_device_get(volatile struct ring_buf *rb, void *el) {
    // caught the head, can't get data
    if (rb->tail == rb->head)
        return -1;
    for (uint32_t i = 0; i < rb->element_size; i++)
        *((uint8_t *)el + i) =
            *((uint8_t *)rb->data_p + rb->element_size * rb->tail + i);
    rb->tail = (rb->tail + 1) % rb->size;
    return 0;
}

/**
 * @brief Init the ring buffer. See `struct ring_buf` for details
 */
inline void rb_init(volatile struct ring_buf *rb, uint32_t size,
                           uint32_t element_size) {
    rb->tail = 0;
    rb->head = 0;
    rb->size = size;
    rb->element_size = element_size;
}

/***********************************************************************************
 * MAILBOX
 ***********************************************************************************/

inline uint32_t mailbox_try_read(volatile struct ring_buf *g_h2a_mbox, uint32_t *buffer) {
    return rb_host_get(g_h2a_mbox, buffer) == 0 ? 1 : 0;
}
inline uint32_t mailbox_read(volatile struct ring_buf *g_h2a_mbox, uint32_t *buffer, uint32_t n_words) {
    int ret;
    while (n_words--) {
        do {
        // If this region is cached and no cache coherency, need a fence
        // asm volatile("fence" : : : "memory");
            ret = rb_host_get(g_h2a_mbox, &buffer[n_words]);
            if (ret) {
                csleep(10);
            }
        } while (ret);
    }
    return 0;
}
inline uint32_t mailbox_write(volatile struct ring_buf *g_a2h_mbox, uint32_t word) {
    int ret;
    do {
        // If this region is cached and no cache coherency, need a fence
        // asm volatile("fence" : : : "memory");
        ret = rb_host_put(g_a2h_mbox, &word);
        if (ret) {
            csleep(10);
        }
    } while (ret);
    return ret;
}