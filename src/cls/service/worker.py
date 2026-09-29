from cls.graph.store import GraphStore
from cls.ingestion.schemas import Trajectory
from cls.learning.analyzer import EpisodeAnalyzer
from cls.learning.extractor import KnowledgeExtractor
from cls.learning.updater import GraphUpdater
from cls.learning.validator import KnowledgeValidator


class LearningWorker:
    """Background pipeline that turns agent trajectories into graph updates."""

    def __init__(self, store: GraphStore):
        self.store = store
        self.analyzer = EpisodeAnalyzer()
        self.extractor = KnowledgeExtractor()
        self.validator = KnowledgeValidator(store)
        self.updater = GraphUpdater(store)

    async def process_trajectory(self, trajectory: Trajectory) -> None:
        """Analyze a trajectory, extract useful procedures, and store them."""
        episode = self.analyzer.analyze(trajectory)
        candidates = self.extractor.extract_candidates(episode)

        for candidate in candidates:
            validated_node = self.validator.validate_and_merge(candidate)
            self.updater.commit_procedure(validated_node)
