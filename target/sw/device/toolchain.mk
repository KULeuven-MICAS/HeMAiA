# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Luca Colagrande <colluca@iis.ee.ethz.ch>

BENDER ?= bender
SNITCH_ROOT = $(shell $(BENDER) path snitch_cluster)
include $(SNITCH_ROOT)/target/snitch_cluster/sw/toolchain.mk