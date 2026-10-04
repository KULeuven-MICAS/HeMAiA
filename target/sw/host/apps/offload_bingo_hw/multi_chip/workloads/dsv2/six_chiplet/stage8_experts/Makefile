# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# DeepSeek-V2-Lite layer 1 on the six-chiplet platform (hemaia_sixchiplet_16MBL3_2cluster: four
# compute chiplets with two clusters each, two memory chiplets, C-M-C rows), stage8_experts.
# The stage's code is main_bingo.py; the build is ../../common/dsv2.mk.
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
WORKLOAD  = dsv2/six_chiplet/stage8_experts
include $(MK_DIR)../../common/dsv2.mk
