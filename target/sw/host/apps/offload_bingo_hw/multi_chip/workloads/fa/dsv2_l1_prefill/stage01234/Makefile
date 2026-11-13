# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
#
# DeepSeek-V2-Lite's layer-1 MLA attention, 256-token prefill, stages 0 to 4 chained; the code is ../../common/fa_stages_mc.
MK_DIR   := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
WORKLOAD  = fa/dsv2_l1_prefill/stage01234
include $(MK_DIR)../../common/fa.mk
