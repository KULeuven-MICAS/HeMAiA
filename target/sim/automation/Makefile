.PHONY: local-ci git-ci sweep-xdma sweep-ara clean

# Number of simulations to run in parallel (override: make local-ci JOBS=4)
JOBS ?= 16

# Which local-CI suite to run. One subdirectory of ci/local_ci/ per tapeout RTL
# cfg; each pairs a cfg with the task list valid for it.
#   tapeout_1c       hemaia_tapeout_1c.hjson       1x versacore
#   tapeout_1c_simd  hemaia_tapeout_1c_simd.hjson  1x versacore+SIMD
#   tapeout_2c       hemaia_tapeout_2c.hjson       2x versacore 256KB
#   tapeout_2c_simd  hemaia_tapeout_2c_simd.hjson  2x versacore 256KB+SIMD
SUITE ?= tapeout_1c

# Local CI: Questasim/VCS regression, run on a host with the EDA tools.
# Source the EDA environment first, or VCS resolves to a version that cannot
# start and every task reports FAIL (0s):
#   source /users/micas/fkong/no_backup/src_hemaia_eda.sh
local-ci:
	cd ci/local_ci/$(SUITE) && python3 run_local_ci.py -j $(JOBS)

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

# Remove the per-run artefacts every flow leaves behind: the staged task_<idx>/
# directories (each holds a copy of the simulation binary, so they are large),
# the Markdown summaries, and __pycache__.
#
# Sweep results (e.g. sweep/ara/ara_cycles.csv, the *_report.md LUTs) are NOT
# removed -- they are the point of running a sweep, and re-deriving one costs
# hours of simulation.
clean:
	rm -rf ci/*/*/task_[0-9]* ci/*/task_[0-9]* sweep/*/task_[0-9]* test/task_[0-9]*
	rm -f  ci/*/*/simulation_summary_*.md ci/*/simulation_summary_*.md \
	       sweep/*/simulation_summary_*.md test/simulation_summary_*.md
	rm -rf ci/*/*/__pycache__ ci/*/__pycache__ sweep/*/__pycache__ test/__pycache__
