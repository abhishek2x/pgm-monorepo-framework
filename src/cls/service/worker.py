import asyncio
import logging
from typing import List

from cls.graph.schema import ProcedureNode
from cls.graph.store import GraphStore
from cls.ingestion.schemas import Trajectory
from cls.ingestion.store import TrajectoryStore
from cls.learning.analyzer import EpisodeAnalyzer
from cls.learning.extractor import KnowledgeExtractor
from cls.learning.updater import GraphUpdater
from cls.learning.validator import KnowledgeValidator

logger = logging.getLogger(__name__)


class LearningWorker:
    """Process durable trajectory jobs and update the procedural graph."""

    def __init__(
        self,
        store: GraphStore,
        trajectory_store: TrajectoryStore = None,
        poll_interval: float = 1.0,
        max_attempts: int = 3,
    ):
        self.store = store
        self.trajectory_store = trajectory_store
        self.poll_interval = poll_interval
        self.max_attempts = max_attempts
        self.analyzer = EpisodeAnalyzer()
        self.extractor = KnowledgeExtractor()
        self.validator = KnowledgeValidator(store)
        self.updater = GraphUpdater(store)

    async def process_trajectory(self, trajectory: Trajectory) -> List[ProcedureNode]:
        """Run the learning pipeline for one trajectory."""
        episode = self.analyzer.analyze(trajectory)
        candidates = self.extractor.extract_candidates(episode)
        committed = []

        for candidate in candidates:
            validated_node = self.validator.validate_and_merge(candidate)
            self.updater.commit_procedure(validated_node)
            committed.append(validated_node)
        return committed

    async def run_once(self) -> bool:
        """Process one queued job; return False when the queue is empty."""
        if self.trajectory_store is None:
            raise RuntimeError("A trajectory store is required to consume queued jobs.")

        trajectory = await asyncio.to_thread(self.trajectory_store.claim_next)
        if trajectory is None:
            return False

        try:
            await self.process_trajectory(trajectory)
        except Exception as error:
            logger.exception("Learning failed for trajectory %s", trajectory.trajectory_id)
            await asyncio.to_thread(
                self.trajectory_store.mark_failed,
                trajectory.trajectory_id,
                str(error),
                self.max_attempts,
            )
        else:
            await asyncio.to_thread(
                self.trajectory_store.mark_completed,
                trajectory.trajectory_id,
            )
        return True

    async def run_forever(self, stop_event: asyncio.Event) -> None:
        """Poll the durable queue until the service begins shutting down."""
        while not stop_event.is_set():
            try:
                await self.run_once()
            except Exception:
                logger.exception("Trajectory worker iteration failed")

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self.poll_interval)
            except asyncio.TimeoutError:
                continue
