# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# Llama-3-8B's whole decode attention block over an 8,192-position cache, rung bcast of the FA decode ablation, its cache copies behind q and W_O pushed during the sweep; the code is
# ../../../llama3/common/llama3_decode_ks_mc.py.
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
WORKLOAD  = fa/llama3_decode_full/bcast
include $(MK_DIR)../../../llama3/common/llama3.mk
