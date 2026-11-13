# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# dsv2/layer1/decode/layer_cluster_xf (cluster-posted weight rings, the root memory chip's other chips' attention rings
# after the input norm's HBM read: params weight_ring.mode cluster, layer.x_load_first): DeepSeek-V2-Lite's layer 1 (MLA
# + MoE) for one decode token over a 1,024-token cache, the attention split by key and every weight multiply on the
# GEMV, the blocks' decode mappings in one graph; the code is ../../../common/dsv2_decode_mc.py (params layer.attn ks,
# moe.gemv), the build the whole-layer one (../../../common/dsv2_prefill.mk: one generator run, every block's sources as
# its dependencies). The host weight prefetcher's build flag: host_kernel_lib.h WEIGHT_PREFETCH_LEAN (the mk's default
# 1, as the MoE block's build).
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
WORKLOAD  = dsv2/layer1/decode/layer_cluster_xf
include $(MK_DIR)../../../common/dsv2_prefill.mk
