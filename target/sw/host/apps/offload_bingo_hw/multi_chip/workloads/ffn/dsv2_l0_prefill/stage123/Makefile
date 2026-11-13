# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# DeepSeek-V2-Lite's dense FFN (layer 0), fresh 256-token prompt, stages 1-3 of one pass: X in L1 before the window; gate|up over the weight stream, dequantise and SwiGLU, the reshape, the quantise and the A8 exchange; the code is main_bingo.py (on ../../common/ffn_prefill_mc).
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
WORKLOAD  = ffn/dsv2_l0_prefill/stage123
include $(MK_DIR)../../common/ffn.mk
