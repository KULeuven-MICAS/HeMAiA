# Fanchen Kong <fanchen.kong@kuleuven.be>
"""Reusable DFG blocks and the linker that assembles them into one graph.

    comm/     COMMON machinery: what a block declares (ports), what it builds through
              (ctx), how two are joined (link), and how an operand that does not match
              gets moved or reshaped (transfer, nest).
    block/    the blocks: FlashAttention, its fold, the MoE FFN.
    verify/   host-side readbacks and comparisons, named by precision.

A block declares its ports and parameters; Pipeline binds them and adds the edges. The
mini-compiler still sees ONE assembled DFG and runs every pass on it exactly as before --
this layer only does assembly, and it never inserts a node, because node creation order is
dispatch order on this machine.

Every port carries SHAPE, LAYOUT, PRECISION and PLACEMENT -- the memory level and, for
L1, which cluster's. All of it is in the signature rather than assumed, because none of it
faults when it is wrong: a mismatched layout computes a scrambled answer, a mismatched
precision reads two elements as one, an address in a memory pool the platform does not
have is simply unmapped, and a remote TCDM handle is written by a transfer that completes
without moving anything.

ONE BLOCK, ONE CLUSTER. A block's kernels run where its cfg says and its ports declare it,
so spreading an operator over the machine is something a layer assembles -- `Scatter`, one
block per cluster, `Gather` -- and the pipeline checks every placement on the way. The two
blocks that do span clusters, FlashAttention and MoeFFN, declare per-cluster ports and say
in their own docstrings why they are one construct rather than several.
"""

# libs reaches the compiler by FLAT module name (bingo_kernel_args, bingo_node, ...), so
# the grouped subdirs have to be on sys.path before any of that resolves. A package has a
# stable location, so it does this for itself rather than making every importer remember.
import os as _os
import sys as _sys

_MC = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _MC not in _sys.path:
    _sys.path.insert(0, _MC)
import _bingo_paths  # noqa: F401,E402

from . import block, comm, verify  # noqa: E402
from .comm import (Block, BlockResult, Cost, Ctx, DType, Layout,  # noqa: E402
                   MemLevel, Pipeline, Port, PortSpec, Ref, Stage, Variant,
                   at_offset, bring_in, check_contract, cluster_of, level_of,
                   link, plan, variants_of)
from .verify import checks  # noqa: E402

__all__ = ["Block", "BlockResult", "Ctx", "Pipeline", "Port", "PortSpec", "Ref", "Stage",
           "Cost", "Variant", "variants_of", "at_offset", "check_contract", "cluster_of",
           "level_of", "link", "DType", "Layout", "MemLevel", "bring_in", "plan",
           "checks", "block", "comm", "verify"]
