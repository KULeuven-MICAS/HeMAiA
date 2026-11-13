# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Llama-3-8B's FFN (synthetic weights), fresh 256-token prompt, stage 0: X of every pass from the root memory chip's HBM into every chip's L3 (its push engine's multicast, a flag a tile) and the first pass's X into every cluster's L1; the code is main_bingo.py (on ../../common/ffn_prefill_mc).
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
WORKLOAD  = ffn/llama3_prefill/stage0
include $(MK_DIR)../../../llama3/common/llama3.mk
