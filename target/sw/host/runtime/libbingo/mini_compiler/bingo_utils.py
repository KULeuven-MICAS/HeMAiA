# Fanchen Kong <fanchen.kong@kuleuven.be>
# Took from zigzag util
import subprocess
import sys
from typing import Any, Generic, Iterator, Literal, Sequence, TypeVar, no_type_check, overload

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import networkx as nx
    from networkx import DiGraph
except ImportError:
    print(f"networkx not found. Installing...")
    install_package("networkx")
    import networkx as nx
    from networkx import DiGraph

T = TypeVar("T")
@no_type_check
class DiGraphWrapper(Generic[T], DiGraph):
    """Wraps the DiGraph class with type annotations for the nodes"""

    @overload
    def in_edges(self, node: T, data: Literal[False]) -> list[tuple[T, T]]: ...

    @overload
    def in_edges(self, node: T, data: Literal[True]) -> list[tuple[T, T, dict[str, Any]]]: ...

    @overload
    def in_edges(self, node: T) -> list[tuple[T, T]]: ...

    def in_edges(  # type: ignore # pylint: disable=W0246
        self,
        node: T,
        data: bool = False,
    ) -> list[tuple[T, T]] | list[tuple[T, T, dict[str, Any]]]:
        return super().in_edges(node, data)  # type: ignore

    @overload
    def out_edges(self, node: T, data: Literal[True]) -> list[tuple[T, T, dict[str, Any]]]: ...

    @overload
    def out_edges(self, node: T, data: Literal[False]) -> list[tuple[T, T]]: ...

    @overload
    def out_edges(self, node: T) -> list[tuple[T, T]]: ...

    def out_edges(  # type: ignore # pylint: disable=W0246
        self,
        node: T,
        data: bool = False,
    ) -> list[tuple[T, T]] | list[tuple[T, T, dict[str, Any]]]:
        return super().out_edges(node, data)  # type: ignore

    @overload
    def in_degree(self, node: Literal[None]) -> Iterator[tuple[T, int]]: ...

    @overload
    def in_degree(self) -> Iterator[tuple[T, int]]: ...

    @overload
    def in_degree(self, node: T) -> int: ...

    def in_degree(self, *args, **kwargs) -> int | Iterator[tuple[T, int]]:  # type: ignore
        return super().in_degree(*args, **kwargs)  # type: ignore

    @overload
    def out_degree(self, node: Literal[None]) -> Iterator[tuple[T, int]]: ...

    @overload
    def out_degree(self) -> Iterator[tuple[T, int]]: ...

    @overload
    def out_degree(self, node: T) -> int: ...

    def out_degree(self, *args, **kwargs) -> int | Iterator[tuple[T, int]]:  # type: ignore
        return super().out_degree(*args, **kwargs)  # type: ignore

    def successors(self, node: T) -> Iterator[T]:  # type: ignore # pylint: disable=W0246
        return super().successors(node)  # type: ignore

    def predecessors(self, node: T) -> Iterator[T]:  # type: ignore # pylint: disable=W0246
        return super().predecessors(node)  # type: ignore

    def topological_sort(self) -> Iterator[T]:
        return nx.topological_sort(self)  # type: ignore

    def add_node(self, node: T) -> None:  # type: ignore # pylint: disable=W0246
        super().add_node(node)  # type: ignore

    def add_nodes_from(self, node: Sequence[T]) -> None:  # pylint: disable=W0246
        super().add_nodes_from(node)  # type: ignore

    def remove_nodes_from(self, nodes: Iterator[T]) -> None:  # pylint: disable=W0246
        super().remove_nodes_from(nodes)  # type: ignore

    def add_edge(self, edge_from: T, edge_to: T) -> None:  # type: ignore # pylint: disable=W0246
        super().add_edge(edge_from, edge_to)  # type: ignore

    def add_edges_from(  # type: ignore # pylint: disable=W0246
        self,
        edges: Sequence[tuple[T, T] | tuple[T, T, Any]],
    ) -> None:
        super().add_edges_from(edges)  # type: ignore

    def all_simple_paths(self, producer: T, consumer: T) -> Iterator[list[T]]:
        return nx.all_simple_paths(self, source=producer, target=consumer)  # type: ignore

    def shortest_path(self, producer: T, consumer: T) -> list[T]:
        return nx.shortest_path(self, producer, consumer)  # type: ignore

    @property
    def node_list(self) -> list[T]:
        return list(self.nodes())  # type: ignore

    def get_node_with_id(self, node_id: int) -> T:
        for node in self.node_list:
            if node.id == node_id:  # type: ignore
                return node
        raise ValueError(f"Node with id {node_id} not found.")