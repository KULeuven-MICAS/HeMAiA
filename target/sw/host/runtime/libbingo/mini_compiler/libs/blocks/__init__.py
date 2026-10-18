# Fanchen Kong <fanchen.kong@kuleuven.be>
"""The blocks themselves -- each a sub-DFG with a declared interface.

ONE BLOCK RUNS ON ONE CLUSTER, and its ports say which. Putting an operator on several
clusters is therefore something a layer WRITES -- one block per cluster, with `Scatter`
and `Gather` at the ends -- rather than something a block hides. The one exception states
its own reason: FlashAttention interleaves its per-cluster pipelines, so the several
clusters are one construct and splitting them would change the program.

Moving a tensor from one cluster's L1 to another's is a block too (`Pull`), for the reason
a Reshape is: the linker may not insert nodes, and a remote L1 handle does not fault. `View`,
`Join` and `After` are the zero-cost relabels that go with it.

A block is imported for its class (FlashAttention) or, where the work is not a block, for
the function that builds it (fa_gather). A Block owns its operands through ports, and a
port names a buffer with a layout and a producer. The attention fold reads each cluster's
softmax arena, a buffer the previous block is still updating in place -- a running state,
not a produced tensor -- so declaring it as a port would promise something the port model
cannot honour.
"""

from . import collective
from .fa_sweep import FaSweepBufs, FaSweepCfg, fa_sweep, fa_sweep_alloc
from .flash_attention import FaCfg, FlashAttention
from .gemm_async import GemmQueue
from .linear import Linear, LinearCfg, LoadStream, WeightRings, record_spec
from .gather import fa_gather
from .mla import (CacheAppend, CacheAppendCfg, MlaAttention, MlaAttnCfg, MlaDimOut, MlaDimPV,
                  MlaDimPVCfg, MlaRowMax, MlaRowMaxCfg, MlaShardCfg, MlaShardOut, MlaShardP,
                  MlaShardPV, MlaShardScores, QAssemble, QAssembleCfg, RopeRows, RopeRowsCfg)
from .move import After, Collect, Fetch, Join, Pull, Stash, View
from .reshape import Reshape, ReshapeCfg
from .route import (MoeRoute, MoeRouteCfg, SlotLoad, SlotLoadCfg, expert_table,
                    pass_record_bytes, record_bytes)
from .shard import Gather, Scatter
from .simd import (AddRow, AddRowCfg, ARowPack, ARowPackCfg, NormCfg, NormRowCfg, QuantARowCfg,
                   Quantize, QuantizeARow, RMSNorm, RMSNormRow, RowCfg,
                   ScaleCols, ScaleColsCfg, ScaleRowBySlot, ScaleRowCfg, SoftmaxRow,
                   SoftmaxRowCfg, SwigluARow, SwigluARowCfg)

__all__ = ["collective", "FaCfg", "FlashAttention", "fa_gather", "GemmQueue",
           "FaSweepBufs", "FaSweepCfg", "fa_sweep", "fa_sweep_alloc", "Gather", "Linear",
           "LinearCfg", "LoadStream", "WeightRings", "record_spec",
           "NormCfg", "Pull", "Quantize", "RMSNorm", "Reshape", "ReshapeCfg",
           "Join", "RowCfg", "Scatter", "View",
           "After", "Collect", "Fetch", "Stash", "ARowPack", "ARowPackCfg", "NormRowCfg", "QuantARowCfg", "QuantizeARow", "RMSNormRow",
           "ScaleCols", "ScaleColsCfg", "AddRow", "AddRowCfg", "ScaleRowBySlot",
           "ScaleRowCfg", "SoftmaxRow", "SoftmaxRowCfg", "SwigluARow", "SwigluARowCfg",
           "CacheAppend", "CacheAppendCfg", "MlaAttention", "MlaAttnCfg", "MlaRowMax",
           "MlaRowMaxCfg", "MlaShardCfg", "MlaShardOut", "MlaShardPV", "MlaShardScores",
           "MlaShardP", "MlaDimPV", "MlaDimPVCfg", "MlaDimOut",
           "QAssemble",
           "QAssembleCfg", "RopeRows", "RopeRowsCfg", "MoeRoute", "MoeRouteCfg",
           "SlotLoad", "SlotLoadCfg", "expert_table", "pass_record_bytes", "record_bytes"]
