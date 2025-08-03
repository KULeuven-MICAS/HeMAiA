// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
extern void dump_mbox(struct ring_buf *rbuf);
extern uint32_t rb_device_put(volatile struct ring_buf *rb, void *el);
extern uint32_t rb_host_get(volatile struct ring_buf *rb, void *el);
extern uint32_t rb_device_get(volatile struct ring_buf *rb, void *el);
extern void rb_init(volatile struct ring_buf *rb, uint32_t size, uint32_t element_size);
extern uint32_t mailbox_try_read(volatile struct ring_buf *g_h2a_mbox, uint32_t *buffer);
extern uint32_t mailbox_read(volatile struct ring_buf *g_h2a_mbox, uint32_t *buffer, uint32_t n_words);
extern uint32_t mailbox_write(volatile struct ring_buf *g_a2h_mbox, uint32_t word);