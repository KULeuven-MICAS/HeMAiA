# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# The FFN strong-scaling sweep, all_n8_cluster rebuilt after the fix that queues the root rings behind X: all_n8 with every weight ring posted by its cluster's DM core (weight_ring.mode cluster) and X's multicast as host graph tasks (hbm_cast.mode posts), the root memory chip's engine queue split 16 + 16, and the D2D yield window: ../../common/ffn_prefill_mc.py and params.hjson.
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
WORKLOAD  = ffn/scaling/all_n8_cluster2
include $(MK_DIR)../../common/ffn.mk
