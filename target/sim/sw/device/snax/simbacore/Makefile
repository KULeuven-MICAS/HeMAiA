# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Xiaoling Yi <xiaoling.yi@esat.kuleuven.be>


MK_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
include $(MK_DIR)/../common.mk

############
## Outputs #
############

OBJS        = $(BUILDDIR)/snax-simbacore-lib.o
ALL_OUTPUTS = $(OBJS)
INCDIRS += $(abspath include)

##########
## Rules #
##########

.PHONY: all
all: $(ALL_OUTPUTS)

.PHONY: clean
clean:
	rm -rf $(BUILDDIR)
