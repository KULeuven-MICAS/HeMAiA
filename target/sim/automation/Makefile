.PHONY: all automated-xdma-maxpool automated-xdma-multicast automated-xdma-transpose automated-xdma-reshape local-ci

# Number of simulations to run in parallel for local-ci (override: make local-ci JOBS=4)
JOBS ?= 16

all: automated-xdma-maxpool automated-xdma-multicast automated-xdma-transpose automated-xdma-reshape

automated-xdma-maxpool:
	cd automated-xdma-maxpool && python3 start.py

automated-xdma-multicast:
	cd automated-xdma-multicast && python3 start.py

automated-xdma-transpose:
	cd automated-xdma-transpose && python3 start.py

automated-xdma-reshape:
	cd automated-xdma-reshape && python3 start.py

local-ci:
	cd ci && python3 start_vsim.py -j $(JOBS)
