# Copyright 2024 KU Leuven
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Yunhao Deng <yunhao.deng@kuleuven.be>
# Fanchen Kong <fanchen.kong@kuleuven.be>
MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR  := $(dir $(MKFILE_PATH))

.PHONY: all clean
all:
	$(MAKE) -C $(MKFILE_DIR)bootrom_chip TARGET=bootrom
	$(MAKE) -C $(MKFILE_DIR)bootrom_sim TARGET=bootrom
# 3. The in-place replacement for the mapped bootrom in the synthesized netlist,
#    generated from the simulation bootrom's binary so the two cannot drift.
	$(MAKE) -C $(MKFILE_DIR)bootrom_netlist_sim bootrom
clean:
	$(MAKE) -C $(MKFILE_DIR)bootrom_chip clean
	$(MAKE) -C $(MKFILE_DIR)bootrom_sim clean
	$(MAKE) -C $(MKFILE_DIR)bootrom_netlist_sim clean