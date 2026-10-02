import hashlib
from datetime import datetime, timezone
from typing import List, Optional
from uuid import uuid4

import networkx as nx
from sqlalchemy import Column, DateTime, JSON, String
from sqlalchemy.orm import declarative_base, sessionmaker

from cls.graph.schema import ProcedureEdge, ProcedureNode

GraphBase = declarative_base()


class ProcedureRecord(GraphBase):
    __tablename__ = "procedure_nodes"

    id = Column(String, primary_key=True)
    data = Column(JSON, nullable=False)


class ProcedureEdgeRecord(GraphBase):
    __tablename__ = "procedure_edges"

    source_id = Column(String, primary_key=True)
    target_id = Column(String, primary_key=True)
    data = Column(JSON, nullable=False)


class RetrievalEventRecord(GraphBase):
    __tablename__ = "retrieval_events"

    id = Column(String, primary_key=True)
    project_id = Column(String, nullable=False)
    task_id = Column(String, nullable=True)
    task_hash = Column(String, nullable=False)
    repository_version = Column(String, nullable=True)
    consumer_user_id = Column(String, nullable=True)
    experiment_run_id = Column(String, nullable=True)
    procedure_ids = Column(JSON, nullable=False)
    created_at = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
    )


class GraphStore:
    """Procedure graph with optional SQL persistence for service deployments."""

    def __init__(self, engine=None):
        self._graph = nx.DiGraph()
        self._engine = engine
        self._session_factory = None
        self._retrieval_events = []
        if engine is not None:
            GraphBase.metadata.create_all(engine)
            self._session_factory = sessionmaker(bind=engine)
            self._load_persisted_graph()

    def add_node(self, node: ProcedureNode) -> None:
        """Persist a node in the graph."""
        data = node.model_dump(mode="json")
        if self._session_factory is not None:
            with self._session_factory() as session:
                record = session.get(ProcedureRecord, node.procedure_id)
                if record is None:
                    session.add(ProcedureRecord(id=node.procedure_id, data=data))
                else:
                    record.data = data
                session.commit()
        self._graph.add_node(node.procedure_id, **data)

    def update_node(self, node: ProcedureNode) -> None:
        """Create or update a node while preserving its identifier."""
        self.add_node(node)

    def add_edge(self, edge: ProcedureEdge) -> None:
        """Connect one procedure to another with a typed relationship."""
        if not self._graph.has_node(edge.source_id):
            raise ValueError(f"Source node {edge.source_id} does not exist.")
        if not self._graph.has_node(edge.target_id):
            raise ValueError(f"Target node {edge.target_id} does not exist.")

        data = edge.model_dump(mode="json")
        if self._session_factory is not None:
            with self._session_factory() as session:
                record = session.get(
                    ProcedureEdgeRecord,
                    (edge.source_id, edge.target_id),
                )
                if record is None:
                    session.add(
                        ProcedureEdgeRecord(
                            source_id=edge.source_id,
                            target_id=edge.target_id,
                            data=data,
                        )
                    )
                else:
                    record.data = data
                session.commit()
        self._graph.add_edge(edge.source_id, edge.target_id, **data)

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

    def record_retrieval_event(
        self,
        project_id: str,
        task_description: str,
        procedure_ids: List[str],
        repository_version: Optional[str] = None,
        consumer_user_id: Optional[str] = None,
        task_id: Optional[str] = None,
        experiment_run_id: Optional[str] = None,
    ) -> str:
        """Record which artifacts were exposed without retaining task prompt text."""
        event_id = str(uuid4())
        event = {
            "id": event_id,
            "project_id": project_id,
            "task_id": task_id,
            "task_hash": hashlib.sha256(task_description.encode("utf-8")).hexdigest(),
            "repository_version": repository_version,
            "consumer_user_id": consumer_user_id,
            "experiment_run_id": experiment_run_id,
            "procedure_ids": list(procedure_ids),
        }
        self._retrieval_events.append(event)

        if self._session_factory is not None:
            with self._session_factory() as session:
                session.add(RetrievalEventRecord(**event))
                session.commit()
        return event_id

    def get_retrieval_events(self) -> List[dict]:
        """Return recorded retrieval exposures for experiment auditing."""
        if self._session_factory is None:
            return list(self._retrieval_events)

        with self._session_factory() as session:
            records = (
                session.query(RetrievalEventRecord)
                .order_by(RetrievalEventRecord.created_at, RetrievalEventRecord.id)
                .all()
            )
            return [
                {
                    "id": record.id,
                    "project_id": record.project_id,
                    "task_id": record.task_id,
                    "task_hash": record.task_hash,
                    "repository_version": record.repository_version,
                    "consumer_user_id": record.consumer_user_id,
                    "experiment_run_id": record.experiment_run_id,
                    "procedure_ids": record.procedure_ids,
                }
                for record in records
            ]

    def _load_persisted_graph(self) -> None:
        with self._session_factory() as session:
            nodes = session.query(ProcedureRecord).all()
            edges = session.query(ProcedureEdgeRecord).all()

        for record in nodes:
            self._graph.add_node(record.id, **record.data)
        for record in edges:
            if self._graph.has_node(record.source_id) and self._graph.has_node(record.target_id):
                self._graph.add_edge(record.source_id, record.target_id, **record.data)
