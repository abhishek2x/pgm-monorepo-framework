from cls.graph.store import GraphStore
from cls.graph.schema import ProcedureNode

class GraphUpdater:
    """
    Updates the graph store with validated procedures.
    """
    def __init__(self, store: GraphStore):
        self.store = store

    def commit_procedure(self, procedure: ProcedureNode) -> None:
        """
        Inserts or updates a procedure in the graph store.
        """
        # If it already exists in the store (same ID), it will update it.
        # Otherwise, it adds a new node.
        self.store.update_node(procedure)
