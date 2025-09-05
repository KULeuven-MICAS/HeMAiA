# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Luca Colagrande <colluca@iis.ee.ethz.ch>

# Add user applications to APPS variable automatically. This is located in the apps directory with depth of 2.

# Now we only compile the snax apps
APPDIR  = apps/snax/snax-bingo-offload
TARGET ?= all

RTDIR    = math
RTDIR   += runtime

.PHONY: all $(RTDIR) $(APPDIR)

all: $(APPDIR)

$(RTDIR):
	$(MAKE) -C $@ $(TARGET)

$(APPDIR): $(RTDIR)
	$(MAKE) -C $@ $(TARGET)
