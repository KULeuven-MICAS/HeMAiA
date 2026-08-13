# Copyright 2024 KU Leuven
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>

MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR  := $(dir $(MKFILE_PATH))
SIM_DIR     := $(MKFILE_DIR)../bootrom_sim

# The netlist stand-in must hold exactly what the simulation boot ROM holds, so it is
# generated from that binary rather than kept in the repo. Build bootrom_sim first:
# its binary is this ROM's only input.
.PHONY: bootrom clean

bootrom: bootrom.v

$(SIM_DIR)/bootrom.bin:
	$(MAKE) -C $(SIM_DIR) bootrom

bootrom.v: $(SIM_DIR)/bootrom.bin $(MKFILE_DIR)gen_bootrom_v.py
	@echo "GEN   <= $(notdir $<)"
	@python3 $(MKFILE_DIR)gen_bootrom_v.py --bin $< --out $@

clean:
	rm -f bootrom.v
