// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
//
// synchronization-latency probe, correctness check; touches only chips 0x00,0x01,0x10,0x11.
// See sync_latency_common/sync_latency_impl.h for the measurement, the round-trip
// rationale and how to read the printed numbers.
#pragma once
#define SYNC_MAX_P 4
#include "sync_latency_impl.h"
