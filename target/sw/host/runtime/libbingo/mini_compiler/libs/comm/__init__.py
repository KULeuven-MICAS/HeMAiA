# Fanchen Kong <fanchen.kong@kuleuven.be>
"""COMMON machinery. Everything a block is built out of, and nothing specific to one.

Two halves, and the split is the interesting part:

    ports / ctx / link   WHAT a block declares and how two of them are joined. A port says
                         four things about an operand -- shape, layout, precision and the
                         memory it lives in -- and the linker binds one block's outputs to
                         the next block's inputs by adding edges, never nodes.

    transfer / nest      HOW a mismatch gets closed when the producer does not already emit
                         what the consumer declared. `transfer` decides what it costs and
                         emits the nodes; `nest` derives the strided xDMA transfer that
                         performs a layout change, and checks its own strides against
                         ground-truth index maps before handing them over.

`paths` is here because both halves need to find the checkout, and there must be exactly
one implementation of that search.
"""

from .ctx import Ctx
from .link import Pipeline, check_contract, link
from .nest import convert_args, index_map
from .paths import add_sim_paths, repo_root
from .ports import (Block, BlockResult, DType, Layout, MemLevel, Port, PortSpec,
                    at_offset, level_of)
from .transfer import Step, bring_in, hoist, plan
from .layout_pass import LayoutPlan, assign_layouts
from .layout_pass import Step as LayoutStep   # transfer.Step is a PLANNED MOVE;
#                                               this one is a STAGE OF A CHAIN.

__all__ = ["Block", "BlockResult", "Ctx", "Pipeline", "Port", "PortSpec",
           "LayoutPlan", "LayoutStep", "assign_layouts",
           "at_offset", "check_contract", "level_of", "link", "DType", "Layout", "MemLevel",
           "bring_in", "hoist", "plan", "Step", "convert_args", "index_map",
           "add_sim_paths", "repo_root"]
