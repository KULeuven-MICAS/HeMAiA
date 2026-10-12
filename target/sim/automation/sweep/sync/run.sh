#!/bin/bash
# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# E3 synchronisation: the figure is re-plotted from measured data; nothing is simulated here.
#   ./run.sh   ->  figs/fig_e3_four_arm.{pdf,png}
set -eu
D=$(cd "$(dirname "$0")" && pwd)
python3 "$D/plot_sync.py"
