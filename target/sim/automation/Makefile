.PHONY: local-ci git-ci sweep-xdma sweep-ara

# Number of simulations to run in parallel (override: make local-ci JOBS=4)
JOBS ?= 16

# Local CI: Questasim/VCS regression, run on a host with the EDA tools.
local-ci:
	cd ci/local_ci && python3 run_local_ci.py -j $(JOBS)

# Git CI: Verilator regression. Normally run by GitHub Actions; run it manually
# only from *inside* the hemaia container with the HW/sim already built.
git-ci:
	cd ci/git_ci && python3 run_git_ci.py -j $(JOBS)

# xDMA op-cost LUT sweep.
sweep-xdma:
	cd sweep/xdma && python3 run_xdma_sweep.py -j $(JOBS)

# Ara FP32 op-cost sweep (decomposed per-kernel apps).
sweep-ara:
	cd sweep/ara && python3 run_ara_sweep.py -j $(JOBS)
