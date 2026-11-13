# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Llama-3-8B's FFN (synthetic weights), fresh 256-token prompt, stages 4-5 of one pass: the chip's A8 in L1 before the window; the down slices over the weight stream, the ChainGathers into the root and the stores; the code is main_bingo.py (on ../../common/ffn_prefill_mc).
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
WORKLOAD  = ffn/llama3_prefill/stage45_async
include $(MK_DIR)../../../llama3/common/llama3.mk
