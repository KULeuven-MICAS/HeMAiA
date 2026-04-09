# Fanchen Kong <fanchen.kong@kuleuven.be>
# The node types and the base class for nodes in the DFG
from abc import ABCMeta
from typing import Literal, Optional
from bingo_kernel_args import BingoKernelArgs

class BingoNode(metaclass=ABCMeta):
    """Abstract base class for nodes in the DFG."""
    def __init__(
        self,
        assigned_chiplet_id: int,
        assigned_cluster_id: int,
        assigned_core_id: int,
        node_name: str = "",
        kernel_name: str = "",
        kernel_args: Optional[BingoKernelArgs] = None,
    ) -> None:
        self._node_id: int = 0
        self._assigned_chiplet_id = assigned_chiplet_id
        self._assigned_cluster_id = assigned_cluster_id
        self._assigned_core_id = assigned_core_id
        self._kernel_name = kernel_name
        self._kernel_args = kernel_args
        self._node_name = node_name if node_name else f"Node_ID{self._node_id}_Chiplet{self._assigned_chiplet_id}_Cluster{self._assigned_cluster_id}_Core{self._assigned_core_id}_Kernel{self._kernel_name}"
        
        # Infer kernel name from args if not provided, but args is provided
        # Or validate them if both are provided (optional, for now just trust user or prioritized args)
        
        self._node_type: Literal['normal', 'dummy', 'gating'] = "normal"
        self._dep_check_enable: bool = False
        self._dep_check_list: list[int] = []
        self._dep_set_enable: bool = False
        self._remote_dep_set_all: bool = False
        self._dep_set_list: list[int] = []
        self._dep_set_chiplet_id: int = 0
        self._dep_set_cluster_id: int = 0
        # DARTS Tier 1: Conditional Execution
        self._cond_exec_en: bool = False
        self._cond_exec_group_id: int = 0
        self._cond_exec_invert: bool = False
        # CERF groups this gating node writes on completion
        self._cerf_write_groups: list[int] = []
        # Set by compiler for auto-inserted gating nodes: points to predecessor
        self._pred_source_node = None
        # Set by compiler for guarded expert nodes: points to gating node (for SW guard)
        self._gating_node = None
        # Set by compiler for guarded expert nodes: index in activation array
        self._cond_node_index: int = None
        # Set by compiler during C emission: C variable name of this node's scratchpad
        self._scratchpad_c_var: str = None

    # Getters and Setters
    @property
    def node_name(self) -> str:
        return self._node_name
    @node_name.setter
    def node_name(self, value: str) -> None:
        self._node_name = value
    @property
    def kernel_name(self) -> str:
        return self._kernel_name

    @kernel_name.setter
    def kernel_name(self, value: str) -> None:
        self._kernel_name = value

    @property
    def kernel_args(self) -> Optional[BingoKernelArgs]:
        return self._kernel_args

    @kernel_args.setter
    def kernel_args(self, value: Optional[BingoKernelArgs]) -> None:
        self._kernel_args = value

    @property
    def node_id(self) -> int:
        return self._node_id

    @node_id.setter
    def node_id(self, value: int) -> None:
        self._node_id = value

    @property
    def assigned_chiplet_id(self) -> int:
        return self._assigned_chiplet_id

    @assigned_chiplet_id.setter
    def assigned_chiplet_id(self, value: int) -> None:
        self._assigned_chiplet_id = value

    @property
    def assigned_cluster_id(self) -> int:
        return self._assigned_cluster_id

    @assigned_cluster_id.setter
    def assigned_cluster_id(self, value: int) -> None:
        self._assigned_cluster_id = value

    @property
    def assigned_core_id(self) -> int:
        return self._assigned_core_id

    @assigned_core_id.setter
    def assigned_core_id(self, value: int) -> None:
        self._assigned_core_id = value

    @property
    def node_type(self) -> Literal['normal', 'dummy', 'gating']:
        return self._node_type

    @node_type.setter
    def node_type(self, value: Literal['normal', 'dummy', 'gating']) -> None:
        self._node_type = value

    @property
    def dep_check_enable(self) -> bool:
        """Get the dep_check_enable flag."""
        return self._dep_check_enable

    @dep_check_enable.setter
    def dep_check_enable(self, value: bool) -> None:
        """Set the dep_check_enable flag."""
        if not isinstance(value, bool):
            raise ValueError("dep_check_enable must be a boolean value.")
        self._dep_check_enable = value

    @property
    def dep_check_list(self) -> list[int]:
        return self._dep_check_list

    @dep_check_list.setter
    def dep_check_list(self, value: list[int]) -> None:
        self._dep_check_list = value

    @property
    def dep_set_enable(self) -> bool:
        """Get the dep_set_enable flag."""
        return self._dep_set_enable

    @dep_set_enable.setter
    def dep_set_enable(self, value: bool) -> None:
        """Set the dep_set_enable flag."""
        if not isinstance(value, bool):
            raise ValueError("dep_set_enable must be a boolean value.")
        self._dep_set_enable = value

    @property
    def remote_dep_set_all(self) -> bool:
        return self._remote_dep_set_all

    @remote_dep_set_all.setter
    def remote_dep_set_all(self, value: bool) -> None:
        self._remote_dep_set_all = value

    @property
    def dep_set_list(self) -> list[int]:
        return self._dep_set_list

    @dep_set_list.setter
    def dep_set_list(self, value: list[int]) -> None:
        self._dep_set_list = value

    @property
    def dep_set_chiplet_id(self) -> int:
        return self._dep_set_chiplet_id

    @dep_set_chiplet_id.setter
    def dep_set_chiplet_id(self, value: int) -> None:
        self._dep_set_chiplet_id = value

    @property
    def dep_set_cluster_id(self) -> int:
        return self._dep_set_cluster_id

    @dep_set_cluster_id.setter
    def dep_set_cluster_id(self, value: int) -> None:
        self._dep_set_cluster_id = value

    @property
    def cond_exec_en(self) -> bool:
        return self._cond_exec_en

    @cond_exec_en.setter
    def cond_exec_en(self, value: bool) -> None:
        self._cond_exec_en = value

    @property
    def cond_exec_group_id(self) -> int:
        return self._cond_exec_group_id

    @cond_exec_group_id.setter
    def cond_exec_group_id(self, value: int) -> None:
        self._cond_exec_group_id = value

    @property
    def cond_exec_invert(self) -> bool:
        return self._cond_exec_invert

    @cond_exec_invert.setter
    def cond_exec_invert(self, value: bool) -> None:
        self._cond_exec_invert = value

    @property
    def cerf_write_groups(self) -> list[int]:
        return self._cerf_write_groups

    @cerf_write_groups.setter
    def cerf_write_groups(self, value: list[int]) -> None:
        self._cerf_write_groups = value

    def __str__(self):
        return f"Node_ID{self._node_id}_Chiplet{self._assigned_chiplet_id}_Cluster{self._assigned_cluster_id}_Core{self._assigned_core_id}_Kernel{self._kernel_name}"

    def list_to_one_hot(lst: list[int]) -> str:
        if (lst==[]):
            return "'0"
        one_hot = 0
        for idx in lst:
            one_hot |= (1 << idx)
        return f"{one_hot}"
    