# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Llama-3-8B's FFN, one pass of 64 tokens with every mechanism: ../../common/ffn_prefill_mc and params.hjson.
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
WORKLOAD  = ffn/llama3_prefill/t64
include $(MK_DIR)../../../llama3/common/llama3.mk
