# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# DeepSeek-V2-Lite layer 1 on the sixteen-chiplet platform (hemaia_sixteenchiplet_16MBL3_2cluster:
# twelve compute chiplets with two clusters each, four memory chiplets on the package edge), full_layer.
# The stage's code is main_bingo.py; the build is ../../common/dsv2.mk.
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
WORKLOAD  = dsv2/sixteen_chiplet/full_layer
include $(MK_DIR)../../common/dsv2.mk
