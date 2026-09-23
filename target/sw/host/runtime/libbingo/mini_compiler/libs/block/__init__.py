# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The blocks themselves -- each a sub-DFG with a declared interface.

ONE BLOCK RUNS ON ONE CLUSTER, and its ports say which. Putting an operator on several
clusters is therefore something a layer WRITES -- one block per cluster, with `Scatter`
and `Gather` at the ends -- rather than something a block hides. The two exceptions state
their own reason: FlashAttention interleaves its per-cluster pipelines, and MoeFFN's
expert lanes are branches of a conditional fork, so in both the several clusters are one
construct and splitting them would change the program.

A block is imported for its class (FlashAttention, MoeFFN) or, where the work is not a
block, for the function that builds it (fa_gather). A Block owns its operands through
ports, and a port names a buffer with a layout and a producer. The attention fold reads
each cluster's softmax arena, a buffer the previous block is still updating in place -- a
running state, not a produced tensor -- so declaring it as a port would promise something
the port model cannot honour.
"""

from .flash_attention import FaCfg, FlashAttention
from .linear import Linear, LinearCfg
from .gather import fa_gather
from .moe import MoeCfg, MoeFFN
from .reshape import Reshape, ReshapeCfg
from .shard import Gather, Scatter
from .simd import (Dequantize, NormCfg, Quantize, Residual, RMSNorm, RoPE,
                   RowCfg)

__all__ = ["FaCfg", "FlashAttention", "fa_gather", "Gather", "Linear", "LinearCfg",
           "MoeCfg", "MoeFFN", "Dequantize", "NormCfg", "Quantize", "RMSNorm", "Reshape",
           "ReshapeCfg", "Residual", "RoPE", "RowCfg", "Scatter"]
