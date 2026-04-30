# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Luca Colagrande <colluca@iis.ee.ethz.ch>

include ../toolchain.mk
IS_CLEAN_GOAL := $(filter clean clean-%,$(MAKECMDGOALS))

ifeq ($(IS_CLEAN_GOAL),)
include $(SNITCH_ROOT)/sw/math/Makefile
else

.PHONY: clean
clean:
	rm -rf build
endif
