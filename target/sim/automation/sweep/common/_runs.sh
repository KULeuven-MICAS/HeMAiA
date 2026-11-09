# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# _runs.sh, sourced by the run tools here: SW (the sweep directory) and where a run's files are.
#   runs_of <tag>  the runs/ directory of run <tag>: RUNS_DIR, or sweep/$EXP/runs (EXP an experiment under sweep/, e.g.
#                  llm_blocks/ffn) when set, else the one sweep/*/*/runs that holds the tag (two holding it is an error)
#   exp_runs       the runs/ directory of RUNS_DIR or EXP, for a tool that takes no run tag
SW=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

runs_of() {
    if [ -n "${RUNS_DIR:-}" ]; then echo "$RUNS_DIR"; return 0; fi
    if [ -n "${EXP:-}" ]; then echo "$SW/$EXP/runs"; return 0; fi
    local d hits=()
    for d in "$SW"/*/*/runs; do
        if [ -e "$d/$1" ] || compgen -G "$d/$1.*" > /dev/null; then hits+=("$d"); fi
    done
    if [ ${#hits[@]} -ne 1 ]; then
        echo "run $1: ${hits[*]:-not found in sweep/*/*/runs}; set EXP=<experiment, e.g. llm_blocks/ffn>" >&2
        return 1
    fi
    echo "${hits[0]}"
}

exp_runs() {
    if [ -n "${RUNS_DIR:-}" ]; then echo "$RUNS_DIR"; return 0; fi
    [ -n "${EXP:-}" ] || { echo "set EXP=<experiment under sweep/, e.g. llm_blocks/ffn> (or RUNS_DIR)" >&2; return 1; }
    echo "$SW/$EXP/runs"
}
