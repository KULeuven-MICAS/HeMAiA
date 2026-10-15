.PHONY: local-ci git-ci sweep-xdma sweep-ara sweep-simd sweep-gemm clean clean-stale

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
# Source the EDA environment first, then run `make local-ci SUITE=...` to run the regression for a given tapeout RTL cfg.
#
# SW_ONLY=1 does a fast re-run that reuses the already-built RTL/bootrom and the
# already-compiled simulation, rebuilding ONLY the per-task app binaries before
# re-running. Handy when iterating on SW: it avoids the ~30-40 min RTL gen + sim
# compile. Requires a prior full `make local-ci SUITE=...` (so the compiled sim
# and generated platform header already exist).
#   make local-ci SUITE=tapeout_1c_simd SW_ONLY=1
SW_ONLY ?=
local-ci:
	cd ci/local_ci/$(SUITE) && python3 run_local_ci.py -j $(JOBS) $(if $(SW_ONLY),--sw-only)

# Git CI: Verilator regression. Normally run by GitHub Actions; run it manually
# only from *inside* the hemaia container with the HW/sim already built.
git-ci:
	cd ci/git_ci && python3 run_git_ci.py -j $(JOBS)

# The sweeps live in sweep/, grouped by what they measure (sweep/README.md): unit_test/ (one
# block's op-cost LUTs), sop_ubench/ (the chiplet package's collectives and synchronisation),
# llm_blocks/ (FFN, MoE and FA on the 16 + 4 chiplet platform), llm_layer/ (a whole layer) and
# misc/. The unit_test LUT sweeps run through these targets; the others have their own run.sh.

# xDMA op-cost LUT sweep.
sweep-xdma:
	cd sweep/unit_test/xdma && python3 run_xdma_sweep.py -j $(JOBS)

# Ara FP32 op-cost sweep (decomposed per-kernel apps).
sweep-ara:
	cd sweep/unit_test/ara && python3 run_ara_sweep.py -j $(JOBS)

# SIMD operator cost LUT sweep.
sweep-simd:
	cd sweep/unit_test/simd && python3 run_simd_sweep.py -j $(JOBS)

# VersaCore GEMM precision sweep.
sweep-gemm:
	cd sweep/unit_test/gemm && python3 run_gemm_sweep.py -j $(JOBS)

# Remove the per-run artefacts every flow leaves behind: the staged task_<idx>/
# directories (each holds a copy of the simulation binary, so they are large),
# the Markdown summaries, and __pycache__.
#
# Sweep results (e.g. sweep/unit_test/ara/ara_cycles.csv, the *_report.md LUTs) are NOT
# removed -- they are the point of running a sweep, and re-deriving one costs
# hours of simulation.
clean:
	rm -rf ci/*/*/task_[0-9]* ci/*/task_[0-9]* sweep/*/*/task_[0-9]* test/task_[0-9]*
	rm -f  ci/*/*/simulation_summary_*.md ci/*/simulation_summary_*.md \
	       sweep/*/*/simulation_summary_*.md test/simulation_summary_*.md
	rm -rf ci/*/*/__pycache__ ci/*/__pycache__ sweep/*/__pycache__ sweep/*/*/__pycache__ \
	       test/__pycache__

# The same artefacts and the bulk of the hsweep experiments (sweep/*/*/runs: core traces after
# their task spans are taken, staged images, simulator copies), old build logs and the workloads'
# generated build/ directories, but only what has been left unchanged for STALE_AGE_H hours, so a
# running simulation is never touched; a run directory holding a file named KEEP never is.
# clean_stale.py lists the rules; `make clean-stale STALE_DRY=1` only lists what would go.
STALE_AGE_H ?= 24
STALE_DRY ?=
clean-stale:
	python3 clean_stale.py --age-h $(STALE_AGE_H) $(if $(STALE_DRY),--dry-run)
