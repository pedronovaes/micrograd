from __future__ import annotations
from typing import List, Set, TYPE_CHECKING
if TYPE_CHECKING: from src.tensor import Value


def build_topological_order(v: Value,
                            topological_order: List,
                            visited: Set) -> List[Value]:
    """
    Topological order from DAG.

    Attributes:
        v: entry node.
        topological_order: an empty list to place DAG nodes as all children are
            visited in the ordering.
        visited: set to maintain all visited nodes.
    """

    if v not in visited:
        visited.add(v)

        for child in v._previous:
            build_topological_order(
                v=child,
                topological_order=topological_order,
                visited=visited
            )

        topological_order.append(v)

    return topological_order
