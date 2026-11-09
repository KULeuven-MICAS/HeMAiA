# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# DeepSeek-V2-Lite layer 1, a fresh 1,024-token prompt, the whole layer (all eight attention chunks); the code is
# ../../../common/dsv2_layer_prefill_mc.py.
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
WORKLOAD  = dsv2/layer1/prefill/full
# each chiplet's compile of this header is large: at most this many at once
PER_CHIP_JOBS = 2
include $(MK_DIR)../../../common/dsv2_prefill.mk
