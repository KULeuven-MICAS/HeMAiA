# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The blocks themselves -- each a sub-DFG with a declared interface.

ONE BLOCK RUNS ON ONE CLUSTER, and its ports say which. Putting an operator on several
clusters is therefore something a layer WRITES -- one block per cluster, with `Scatter`
and `Gather` at the ends -- rather than something a block hides. The two exceptions state
their own reason: FlashAttention interleaves its per-cluster pipelines, and MoeFFN's
expert lanes are branches of a conditional fork, so in both the several clusters are one
construct and splitting them would change the program.

Moving a tensor from one cluster's L1 to another's is a block too (`Broadcast`, `Pull`),
for the reason a Reshape is: the linker may not insert nodes, and a remote L1 handle does
not fault. `Slice` and `TransposeView` are the zero-cost views that go with them.

A block is imported for its class (FlashAttention, MoeFFN) or, where the work is not a
block, for the function that builds it (fa_gather). A Block owns its operands through
ports, and a port names a buffer with a layout and a producer. The attention fold reads
each cluster's softmax arena, a buffer the previous block is still updating in place -- a
running state, not a produced tensor -- so declaring it as a port would promise something
the port model cannot honour.
"""

from . import collective
from .flash_attention import FaCfg, FlashAttention
from .linear import Linear, LinearCfg, LoadStream, WeightRings, record_spec
from .gather import fa_gather
from .mla import (CacheAppend, CacheAppendCfg, MlaAttention, MlaAttnCfg, MlaDimOut, MlaDimPV,
                  MlaDimPVCfg, MlaRowMax, MlaRowMaxCfg, MlaShardCfg, MlaShardOut, MlaShardP,
                  MlaShardPV, MlaShardScores, QAssemble, QAssembleCfg, RopeRows, RopeRowsCfg)
from .moe import MoeCfg, MoeFFN
from .move import (After, Broadcast, Collect, Fetch, Join, MoveCfg, Pull, Slice, Stash,
                   TransposeView, View)
from .reshape import Reshape, ReshapeCfg
from .route import (MoeRoute, MoeRouteCfg, SlotLoad, SlotLoadCfg, expert_table,
                    pass_record_bytes, record_bytes)
from .shard import Gather, Scatter
from .simd import (AddRow, AddRowCfg, ARowLoad, ARowLoadCfg, ARowPack, ARowPackCfg, Dequantize, NormCfg, NormRowCfg, QuantARowCfg,
                   Quantize, QuantizeARow, Residual, RMSNorm, RMSNormRow, RoPE, RowCfg,
                   ScaleCols, ScaleColsCfg, ScaleRowBySlot, ScaleRowCfg, SoftmaxRow,
                   SoftmaxRowCfg, SwigluARow, SwigluARowCfg)

__all__ = ["collective", "Broadcast", "FaCfg", "FlashAttention", "fa_gather", "Gather", "Linear",
           "LinearCfg", "LoadStream", "WeightRings", "record_spec", "MoeCfg", "MoeFFN", "MoveCfg",
           "Dequantize", "NormCfg", "Pull", "Quantize", "RMSNorm", "Reshape", "ReshapeCfg",
           "Join", "Residual", "RoPE", "RowCfg", "Scatter", "Slice", "TransposeView", "View",
           "After", "Collect", "Fetch", "Stash", "ARowLoad", "ARowLoadCfg", "ARowPack", "ARowPackCfg", "NormRowCfg", "QuantARowCfg", "QuantizeARow", "RMSNormRow",
           "ScaleCols", "ScaleColsCfg", "AddRow", "AddRowCfg", "ScaleRowBySlot",
           "ScaleRowCfg", "SoftmaxRow", "SoftmaxRowCfg", "SwigluARow", "SwigluARowCfg",
           "CacheAppend", "CacheAppendCfg", "MlaAttention", "MlaAttnCfg", "MlaRowMax",
           "MlaRowMaxCfg", "MlaShardCfg", "MlaShardOut", "MlaShardPV", "MlaShardScores",
           "MlaShardP", "MlaDimPV", "MlaDimPVCfg", "MlaDimOut",
           "QAssemble",
           "QAssembleCfg", "RopeRows", "RopeRowsCfg", "MoeRoute", "MoeRouteCfg",
           "SlotLoad", "SlotLoadCfg", "expert_table", "pass_record_bytes", "record_bytes"]
