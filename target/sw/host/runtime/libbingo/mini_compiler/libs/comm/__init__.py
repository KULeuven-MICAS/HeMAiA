# Fanchen Kong <fanchen.kong@kuleuven.be>
"""COMMON machinery. Everything a block is built out of, and nothing specific to one.

Two halves, and the split is the interesting part:

    ports / ctx / link   WHAT a block declares and how two of them are joined. A port
                         states an operand's shape, layout, precision and PLACEMENT -- the
                         memory level and, for L1, which cluster's -- and the linker binds
                         one block's outputs to the next block's inputs by adding edges,
                         never nodes.

    variant              WHAT a block could equally well have been. A block is a FAMILY of
                         realisations, not one pinned signature; `link` connects the chain
                         first, resolves every boundary against those families, and only
                         then asks each block to emit its own sub-DFG.

    transfer / nest      WHAT A BLOCK MOVES ITS OWN OPERANDS WITH. Nothing here is reached
                         by the linker: a gap between two blocks is a refusal, and these
                         are what a block calls INSIDE its own build() to fetch what it
                         declared. `transfer` prices a move and emits it; `nest` derives
                         the strided xDMA transfer that performs a layout change, and
                         checks its own strides against ground-truth index maps before
                         handing them over.

`paths` is here because both halves need to find the checkout, and there must be exactly
one implementation of that search.
"""

from .ctx import Ctx
from .link import Pipeline, Ref, Stage, check_contract, link
from .nest import convert_args, index_map
from .paths import add_sim_paths, repo_root
from .ports import (Block, BlockResult, DType, Layout, MemLevel, Port, PortSpec,
                    at_offset, cluster_of, level_of)
from .transfer import bring_in, plan
from .variant import Cost, Variant, cost_of, free_fields, variants_of

__all__ = ["Block", "BlockResult", "Ctx", "Pipeline", "Port", "PortSpec", "Ref", "Stage",
           "Cost", "Variant", "cost_of", "variants_of", "free_fields",
           "at_offset", "check_contract", "cluster_of", "level_of", "link",
           "DType", "Layout", "MemLevel", "bring_in", "plan", "convert_args", "index_map",
           "add_sim_paths", "repo_root"]
