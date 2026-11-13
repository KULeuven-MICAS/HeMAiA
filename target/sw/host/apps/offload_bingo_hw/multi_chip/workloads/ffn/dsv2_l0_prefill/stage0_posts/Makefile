# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# DeepSeek-V2-Lite's dense FFN (layer 0), fresh 256-token prompt, stage 0: X of every pass from the root memory chip's HBM into every chip's L3 (its push engine's multicast, a flag a tile) and the first pass's X into every cluster's L1; the root's host posts each tile to the push engine as a task of its own (hbm_cast.mode posts); the code is main_bingo.py (on ../../common/ffn_prefill_mc.py).
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
WORKLOAD  = ffn/dsv2_l0_prefill/stage0_posts
include $(MK_DIR)../../common/ffn.mk
