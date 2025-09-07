# Copyright 2023 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Luca Colagrande <colluca@iis.ee.ethz.ch>

# Add user applications to APPS variable automatically. This is located in the apps directory with depth of 2.

# Those apps are standalone apps
# Will be gradually moved to the kernels
APPDIR  = apps/snax/snax-test-integration
APPDIR += apps/snax/snax-xdma-maxpool
APPDIR += apps/snax/snax-xdma-memset
APPDIR += apps/snax/snax-xdma-multicast
APPDIR += apps/snax/snax-xdma-copy
APPDIR += apps/snax/snax-xdma-hazard
APPDIR += apps/snax/snax-xdma-transpose
APPDIR += apps/snax/snax-xdma-local-transpose
APPDIR += apps/snax/snax-xdma-reshape
APPDIR += apps/snax/snax-xdma-multicast-exploration
APPDIR += apps/snax/snax-xdma-overhead
APPDIR += apps/snax/snax-axi-torture
APPDIR += apps/snax/snax-axi-torture-devil
APPDIR += apps/snax/snax-xdma-reshape-multicast
# In the future we only need this offload bingo app
APPDIR += apps/snax/snax-bingo-offload

TARGET ?= all

RTDIR    = math
RTDIR   += runtime

.PHONY: all $(RTDIR) $(APPDIR)

all: $(APPDIR)

$(RTDIR):
	$(MAKE) -C $@ $(TARGET)

$(APPDIR): $(RTDIR)
	$(MAKE) -C $@ $(TARGET)
