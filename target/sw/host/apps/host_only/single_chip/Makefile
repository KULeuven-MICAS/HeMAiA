# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>

# Single-chip host-only applications
SUBDIRS = hello_world
SUBDIRS += clk_rst_configurator
SUBDIRS += d2d_link_configurator
TARGET ?= all

.PHONY: all clean partial-build finalize-build $(SUBDIRS)

all partial-build finalize-build: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ $(TARGET)

clean:
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir clean; \
	done
