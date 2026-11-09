# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# The FFN strong-scaling sweep, all at 4 chips: ../../common/ffn_prefill_mc.py and params.hjson.
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
WORKLOAD  = ffn/sweep_t64/all_n4
include $(MK_DIR)../../common/ffn.mk
