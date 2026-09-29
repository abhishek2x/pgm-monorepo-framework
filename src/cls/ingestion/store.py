from sqlalchemy import Column, String, Integer, Float, JSON, DateTime, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

Base = declarative_base()

class TrajectoryRecord(Base):
    __tablename__ = 'trajectories'

    id = Column(String, primary_key=True) # Could be a UUID based on user_id + task_id + timestamp
    user_id = Column(String, nullable=False)
    agent_id = Column(String, nullable=False)
    project_id = Column(String, nullable=False)
    task_id = Column(String, nullable=False)
    repository = Column(String, nullable=False)
    repository_commit = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Store complex nested data as JSON for the MVP
    messages = Column(JSON, default=list)
    steps = Column(JSON, default=list)
    files_touched = Column(JSON, default=list)
    errors = Column(JSON, default=list)
    
    outcome = Column(String, nullable=False)
    token_usage = Column(JSON, default=dict)
    latency = Column(Float, nullable=False)

# Setup basic sqlite engine for MVP
def get_engine(db_path="sqlite:///trajectories.db"):
    engine = create_engine(db_path)
    Base.metadata.create_all(engine)
    return engine

def get_session(engine):
    Session = sessionmaker(bind=engine)
    return Session()
