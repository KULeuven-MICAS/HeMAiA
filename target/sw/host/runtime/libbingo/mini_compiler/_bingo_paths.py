# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Make the grouped mini_compiler subdirs importable with flat module names.

The compiler used to be twenty files in one directory, which gave no hint about what runs
when. They are now grouped by ROLE:

    graph/      the DFG data model -- nodes, handles, the graph itself
    kernels/    the kernel ABI: one args class per kernel, matching the C structs
    passes/     what runs OVER an assembled graph: validate, transform, allocate dep tags,
                place L1, emit C, report. Plus the analyses they use (liveness, the
                packer, the hardware-manager model).
    platform/   the machine: core roles, cluster counts, transfer-size rules
    libs/       reusable blocks built on top of all of it, and their linker
    tests/

Consumers still import them FLAT (``from bingo_dfg import BingoDFG``), because sixty-odd
workloads do and the grouping is not worth breaking them. Importing this module -- once
``mini_compiler`` itself is on ``sys.path`` -- appends the subdirs, which is what makes
those flat imports resolve. Every main_bingo.py that uses the compiler imports it first.

This is the same arrangement, for the same reason, as ``util/sim/_usg_paths.py``.
"""

import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))

for _s in ("graph", "kernels", "passes", "platform"):
    _sub = os.path.join(_THIS, _s)
    if _sub not in sys.path:
        sys.path.append(_sub)
# libs is a real package and is imported as one, so the mini_compiler directory itself
# has to stay on the path. It normally already is -- this makes it so when it is not.
if _THIS not in sys.path:
    sys.path.append(_THIS)


def repo_root():
    """The HeMAiA checkout root, found by a MARKER rather than by counting `..`.

    A fixed number of parent steps is correct for exactly one directory depth and silent
    when it is wrong: the computed path simply does not exist, and the failure surfaces
    much later as a missing generated header, from a module that never mentions paths.
    Grouping these files into subdirs moved two such counts by one level, which is how
    this function came to exist. bingo_sim_check._hemaia_root() already did it this way.
    """
    d = _THIS
    for _ in range(12):
        if os.path.isfile(os.path.join(d, "Bender.yml")) and \
                os.path.isdir(os.path.join(d, "util", "sim")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    raise RuntimeError(
        f"no HeMAiA checkout above {_THIS}: looked for a directory holding both "
        f"Bender.yml and util/sim.")


def target_sw():
    """<root>/target/sw -- where the generated platform and device headers live."""
    return os.path.join(repo_root(), "target", "sw")
