# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# The FFN ablation ladder at 64 tokens, rung all: ../../common/ffn_prefill_mc (README.md) and params.hjson.
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
WORKLOAD  = ffn/dsv2_l0_prefill_ablation/all
include $(MK_DIR)../../common/ffn.mk
