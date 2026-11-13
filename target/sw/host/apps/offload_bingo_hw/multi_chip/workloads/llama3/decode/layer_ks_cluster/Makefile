# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Llama-3-8B (synthetic weights), the decode layer with cluster-posted weight rings and host HBM-cast posts, the attention split by key and every FFN weight multiply on the GEMV (input RMSNorm, the attention, the glue, the FFN, the final residual), one token over a 1,024-position KV cache, x from and the result into HBM; the
# code is ../../common/llama3_decode_layer_mc.py.
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
WORKLOAD  = llama3/decode/layer_ks_cluster
include $(MK_DIR)../../common/llama3.mk
