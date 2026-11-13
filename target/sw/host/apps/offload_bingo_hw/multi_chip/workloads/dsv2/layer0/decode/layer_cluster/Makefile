# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# dsv2/layer0/decode/layer_cluster (cluster-posted weight rings: params weight_ring.mode cluster): DeepSeek-V2-Lite's
# layer 0 (MLA + the dense FFN) for one decode token over a 1,024-token cache, the attention split by key and every
# weight multiply on the GEMV, the blocks' decode mappings in one graph (the attention is layer 1's: params.hjson); the
# code is ../../../common/dsv2_decode_mc.py (params layer.attn ks, ffn.gu_gemv), the build the whole-layer one
# (../../../common/dsv2_prefill.mk). The host weight prefetcher's build flag: host_kernel_lib.h WEIGHT_PREFETCH_LEAN 0,
# as both blocks' builds.
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
WORKLOAD  = dsv2/layer0/decode/layer_cluster
PREFETCH_LEAN = 0
include $(MK_DIR)../../../common/dsv2_prefill.mk
