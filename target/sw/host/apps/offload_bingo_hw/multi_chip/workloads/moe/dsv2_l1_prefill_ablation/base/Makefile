# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# DeepSeek-V2-Lite's layer-1 MoE prefill, the ablation's rung base; the code is
# ../../common/moe_prefill_mc/.
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
WORKLOAD  = moe/dsv2_l1_prefill_ablation/base
include $(MK_DIR)../../common/moe.mk
