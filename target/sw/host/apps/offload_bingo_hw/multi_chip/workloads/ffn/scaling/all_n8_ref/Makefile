# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# The FFN strong-scaling sweep, all_n8 with the D2D yield window (the reference for all_n8_cluster): ../../common/ffn_prefill_mc.py and params.hjson.
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
WORKLOAD  = ffn/scaling/all_n8_ref
include $(MK_DIR)../../common/ffn.mk
