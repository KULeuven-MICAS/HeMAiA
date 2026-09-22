# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The blocks themselves -- each a sub-DFG with a declared interface.

A block is imported for its class (FlashAttention, MoeFFN) or, where the work is not a
block,
for the function that builds it (fa_gather). That distinction is deliberate and not a
tidiness failure: a Block owns its operands through ports, and a port names a buffer with
a layout and a producer. The attention fold reads each cluster's softmax arena, a buffer
the previous block is still updating in place -- a running state, not a produced tensor --
so declaring it as a port would promise something the port model cannot honour.
"""

from .flash_attention import FaCfg, FlashAttention
from .gather import fa_gather
from .moe import MoeCfg, MoeFFN

__all__ = ["FaCfg", "FlashAttention", "fa_gather", "MoeCfg", "MoeFFN"]
