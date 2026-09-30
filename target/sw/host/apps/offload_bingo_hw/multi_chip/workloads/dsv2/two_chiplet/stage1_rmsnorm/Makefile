# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# DeepSeek-V2-Lite layer 1 on the two-chiplet platform (hemaia_twochiplet_16MBL3_4cluster: one
# compute chiplet with four clusters, and the memory chiplet with its HBM), stage 1: x (HBM) -> RMSNorm + quantiser -> x8.
# The stage's code is main_bingo.py; the build is ../../common/dsv2.mk.
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
WORKLOAD  = dsv2/two_chiplet/stage1_rmsnorm
include $(MK_DIR)../../common/dsv2.mk
