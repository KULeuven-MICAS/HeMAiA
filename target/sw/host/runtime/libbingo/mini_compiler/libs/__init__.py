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

Every port carries four things: SHAPE, LAYOUT, PRECISION and LOCATION. All four are in the
signature rather than assumed, because none of them faults when it is wrong -- a mismatched
layout computes a scrambled answer, a mismatched precision reads two elements as one, and
an address in a memory pool the platform does not have is simply unmapped.
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
from .comm import (Block, BlockResult, Ctx, DType, Layout, MemLevel,  # noqa: E402
                   Pipeline, Port, PortSpec, at_offset, bring_in, check_contract,
                   level_of, link, plan)
from .verify import checks  # noqa: E402

__all__ = ["Block", "BlockResult", "Ctx", "Pipeline", "Port", "PortSpec",
           "at_offset", "check_contract", "level_of", "link", "DType", "Layout", "MemLevel",
           "bring_in", "plan", "checks", "block", "comm", "verify"]
