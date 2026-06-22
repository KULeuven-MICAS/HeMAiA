// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Fanchen Kong <fanchen.kong@kuleuven.be>
//
// Multi-precision cycle sweep + correctness check for the Ara host kernel
// "reduce_mean". Precision is passed to the dispatcher __host_bingo_kernel_reduce_mean as a
// runtime arg word and swept here (fp32/fp16 for all float ops; +int8/int16 for
// integer-meaningful ops). See target/sw/host/runtime/ara_sweep.h.
#include "host.h"
#include "host_kernel_lib.h"
#include "op_test_data.h"
#include "ara_sweep.h"

ARA_MAIN_REDUCE_FLOAT(reduce_mean, 0.001f)
