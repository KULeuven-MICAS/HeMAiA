// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
//

extern void snrt_xchip_writew(uint8_t chip_id, uint32_t addr_lo, uint32_t val);

extern uint32_t snrt_xchip_readw(uint8_t chip_id, uint32_t addr_lo);

extern void snrt_xchip_writeb(uint8_t chip_id, uint32_t addr_lo, uint8_t val);

extern uint8_t snrt_xchip_readb(uint8_t chip_id, uint32_t addr_lo);
