# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Usage of absolute paths is required to externally include this Makefile
COLL_DATA_MK_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
COLL_DATA_DIR    := $(realpath $(COLL_DATA_MK_DIR))

DATA_CFG ?= $(COLL_DATA_DIR)/params.hjson

DATA_H = $(COLL_DATA_DIR)/data.h

$(DATA_H): $(COLL_DATA_DIR)/datagen.py $(DATA_CFG)
	python3 $< -c $(DATA_CFG) > $@.tmp && mv -f $@.tmp $@

.PHONY: clean-data clean

clean-data:
	rm -f $(DATA_H)

clean: clean-data
