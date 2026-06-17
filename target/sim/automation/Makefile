.PHONY: local-ci

# Number of simulations to run in parallel for local-ci (override: make local-ci JOBS=4)
JOBS ?= 16

local-ci:
	cd ci && python3 start_vsim.py -j $(JOBS)
