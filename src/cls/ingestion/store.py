from datetime import datetime, timedelta, timezone

from sqlalchemy import Column, DateTime, Float, Integer, JSON, String, create_engine, update
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class TrajectoryRecord(Base):
    __tablename__ = 'trajectories'

    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    agent_id = Column(String, nullable=False)
    project_id = Column(String, nullable=False)
    task_id = Column(String, nullable=False)
    repository = Column(String, nullable=False)
    repository_commit = Column(String, nullable=False)
    timestamp = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
    )
    
    # Store complex nested data as JSON for the MVP
    messages = Column(JSON, default=list)
    steps = Column(JSON, default=list)
    files_touched = Column(JSON, default=list)
    errors = Column(JSON, default=list)
    
    outcome = Column(String, nullable=False)
    token_usage = Column(JSON, default=dict)
    latency = Column(Float, nullable=False)


class TrajectoryJobRecord(Base):
    __tablename__ = "trajectory_jobs"

    id = Column(String, primary_key=True)
    payload = Column(JSON, nullable=False)
    status = Column(String, nullable=False, default="pending")
    attempt_count = Column(Integer, nullable=False, default=0)
    last_error = Column(String, nullable=True)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    updated_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))


class TrajectoryStore:
    """Persist trajectories as idempotent jobs for the learning worker."""

    def __init__(self, engine):
        self.engine = engine
        self._session_factory = sessionmaker(bind=engine)
        Base.metadata.create_all(engine)

    def enqueue(self, trajectory) -> bool:
        """Store a trajectory once; return False when its ID was already queued."""
        payload = trajectory.model_dump(mode="json")
        with self._session_factory() as session:
            existing = session.get(TrajectoryJobRecord, trajectory.trajectory_id)
            if existing is not None:
                if existing.payload != payload:
                    raise ValueError(
                        f"Trajectory ID {trajectory.trajectory_id} already exists with different content."
                    )
                return False
            session.add(
                TrajectoryRecord(
                    id=trajectory.trajectory_id,
                    user_id=trajectory.user_id,
                    agent_id=trajectory.agent_id,
                    project_id=trajectory.project_id,
                    task_id=trajectory.task_id,
                    repository=trajectory.repository,
                    repository_commit=trajectory.repository_commit,
                    timestamp=trajectory.timestamp.astimezone(timezone.utc).replace(tzinfo=None),
                    messages=payload["messages"],
                    steps=payload["steps"],
                    files_touched=list(trajectory.files_touched),
                    errors=list(trajectory.errors),
                    outcome=trajectory.outcome,
                    token_usage=dict(trajectory.token_usage),
                    latency=trajectory.latency,
                )
            )
            session.add(
                TrajectoryJobRecord(
                    id=trajectory.trajectory_id,
                    payload=payload,
                    status="pending",
                )
            )
            try:
                session.commit()
            except Exception:
                session.rollback()
                with self._session_factory() as check_session:
                    existing = check_session.get(TrajectoryJobRecord, trajectory.trajectory_id)
                    if existing is not None:
                        if existing.payload != payload:
                            raise ValueError(
                                f"Trajectory ID {trajectory.trajectory_id} already exists with different content."
                            )
                        return False
                raise
        return True

    def get_trajectory(self, trajectory_id: str):
        """Load the immutable trajectory payload by its stable ID."""
        with self._session_factory() as session:
            record = session.get(TrajectoryJobRecord, trajectory_id)
            payload = record.payload if record is not None else None
        if payload is None:
            return None

        from cls.ingestion.schemas import Trajectory

        return Trajectory.model_validate(payload)

    def claim_next(self):
        """Atomically claim the oldest pending trajectory, if one is available."""
        with self._session_factory() as session:
            record = (
                session.query(TrajectoryJobRecord)
                .filter(TrajectoryJobRecord.status == "pending")
                .order_by(TrajectoryJobRecord.created_at, TrajectoryJobRecord.id)
                .first()
            )
            if record is None:
                return None

            result = session.execute(
                update(TrajectoryJobRecord)
                .where(
                    TrajectoryJobRecord.id == record.id,
                    TrajectoryJobRecord.status == "pending",
                )
                .values(
                    status="processing",
                    attempt_count=TrajectoryJobRecord.attempt_count + 1,
                    updated_at=datetime.now(timezone.utc).replace(tzinfo=None),
                )
            )
            if result.rowcount != 1:
                session.rollback()
                return None

            payload = record.payload
            session.commit()

        from cls.ingestion.schemas import Trajectory

        return Trajectory.model_validate(payload)

    def mark_completed(self, trajectory_id: str) -> None:
        """Mark a claimed trajectory as processed."""
        self._update_status(trajectory_id, "completed", None)

    def mark_failed(self, trajectory_id: str, error: str, max_attempts: int = 3) -> None:
        """Retry a failed job until its configured attempt limit is reached."""
        with self._session_factory() as session:
            record = session.get(TrajectoryJobRecord, trajectory_id)
            if record is None or record.status != "processing":
                raise ValueError(f"Trajectory {trajectory_id} is not being processed.")
            record.status = "failed" if record.attempt_count >= max_attempts else "pending"
            record.last_error = error[:4000]
            record.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
            session.commit()

    def get_status(self, trajectory_id: str):
        with self._session_factory() as session:
            record = session.get(TrajectoryJobRecord, trajectory_id)
            return record.status if record is not None else None

    def recover_interrupted(self, stale_after_seconds: int = 300) -> int:
        """Return old processing jobs to the queue after an unclean shutdown."""
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(
            seconds=stale_after_seconds
        )
        with self._session_factory() as session:
            result = session.execute(
                update(TrajectoryJobRecord)
                .where(
                    TrajectoryJobRecord.status == "processing",
                    TrajectoryJobRecord.updated_at < cutoff,
                )
                .values(status="pending", updated_at=datetime.now(timezone.utc).replace(tzinfo=None))
            )
            session.commit()
            return result.rowcount or 0

    def _update_status(self, trajectory_id: str, status: str, error):
        with self._session_factory() as session:
            record = session.get(TrajectoryJobRecord, trajectory_id)
            if record is None or record.status != "processing":
                raise ValueError(f"Trajectory {trajectory_id} is not being processed.")
            record.status = status
            record.last_error = error
            record.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
            session.commit()


def get_engine(db_path="sqlite:///trajectories.db"):
    connect_args = {"check_same_thread": False} if db_path.startswith("sqlite") else {}
    engine = create_engine(db_path, connect_args=connect_args)
    Base.metadata.create_all(engine)
    return engine

def get_session(engine):
    Session = sessionmaker(bind=engine)
    return Session()
