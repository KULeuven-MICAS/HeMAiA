# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# The FFN ablation ladder, rung bcast_reduce; ../../common/ffn_prefill_mc.py (THE ABLATION) and params.hjson.
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
WORKLOAD  = ffn/dsv2_l0_prefill_ablation/bcast_reduce
include $(MK_DIR)../../common/ffn.mk
