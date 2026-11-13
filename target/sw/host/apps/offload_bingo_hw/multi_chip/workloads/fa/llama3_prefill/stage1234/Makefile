# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Llama-3-8B's attention (synthetic weights), fresh 256-token prompt, stages 1-4 alone (x, the projections, the K/V trade); the code is
# main_bingo.py.
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
WORKLOAD  = fa/llama3_prefill/stage1234
include $(MK_DIR)../../../llama3/common/llama3.mk
