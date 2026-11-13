# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# DeepSeek-V2-Lite's dense FFN (layer 0), fresh 256-token prompt, the whole block, every pass, X from HBM; the code is ../../common/ffn_prefill_mc.
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
WORKLOAD  = ffn/dsv2_l0_prefill/full
include $(MK_DIR)../../common/ffn.mk
