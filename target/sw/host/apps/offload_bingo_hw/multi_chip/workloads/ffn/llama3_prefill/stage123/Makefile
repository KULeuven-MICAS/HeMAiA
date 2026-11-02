# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Llama-3-8B's FFN (synthetic weights), fresh 256-token prompt, stages 1-3 of one pass: X in L1 before the window; gate|up over the weight stream, dequantise and SwiGLU, the reshape, the quantise and the A8 exchange; the code is ../../common/ffn_stages_mc.py.
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
WORKLOAD  = ffn/llama3_prefill/stage123
include $(MK_DIR)../../../llama3/common/llama3.mk
