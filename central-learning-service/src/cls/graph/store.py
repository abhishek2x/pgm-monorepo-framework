from typing import List, Optional

import networkx as nx

from cls.graph.schema import ProcedureEdge, ProcedureNode


class GraphStore:
    """Small in-memory graph used to store validated procedures."""

    def __init__(self):
        self._graph = nx.DiGraph()

    def add_node(self, node: ProcedureNode) -> None:
        """Persist a node in the graph."""
        self._graph.add_node(node.procedure_id, **node.model_dump())

    def update_node(self, node: ProcedureNode) -> None:
        """Create or update a node while preserving its identifier."""
        if self._graph.has_node(node.procedure_id):
            nx.set_node_attributes(self._graph, {node.procedure_id: node.model_dump()})
            return
        self.add_node(node)

    def add_edge(self, edge: ProcedureEdge) -> None:
        """Connect one procedure to another with a typed relationship."""
        if not self._graph.has_node(edge.source_id):
            raise ValueError(f"Source node {edge.source_id} does not exist.")
        if not self._graph.has_node(edge.target_id):
            raise ValueError(f"Target node {edge.target_id} does not exist.")

        self._graph.add_edge(edge.source_id, edge.target_id, **edge.model_dump())

    def get_node(self, procedure_id: str) -> Optional[ProcedureNode]:
        """Retrieve a single node by its identifier."""
        if self._graph.has_node(procedure_id):
            return ProcedureNode(**self._graph.nodes[procedure_id])
        return None

    def get_all_nodes(self) -> List[ProcedureNode]:
        """Return all nodes in the graph."""
        return [ProcedureNode(**data) for _, data in self._graph.nodes(data=True)]

    def get_edges(self, source_id: str) -> List[ProcedureEdge]:
        """Return all outgoing edges from a single source node."""
        if not self._graph.has_node(source_id):
            return []

        edges = []
        for _, _, data in self._graph.out_edges(source_id, data=True):
            edges.append(ProcedureEdge(**data))
        return edges

    def get_subgraph_by_scope(self, scope: str) -> List[ProcedureNode]:
        """Return every procedure that belongs to the given project scope."""
        return [
            ProcedureNode(**data)
            for _, data in self._graph.nodes(data=True)
            if data.get("scope") == scope
        ]
