from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, ForeignKey, Boolean # lets us write Python instead of raw SQL to talk to the database
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

# Connect to PostgreSQL
engine = create_engine(os.getenv("DATABASE_URL"))
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# TABLE 1 - Learning Goals
class LearningGoal(Base):
    __tablename__ = "learning_goals"
    
    id = Column(Integer, primary_key=True)
    goal = Column(String)
    topics = Column(Text)  # stored as comma separated
    hours_per_day = Column(Integer)
    start_date = Column(DateTime, default=datetime.now)
    target_end_date = Column(DateTime)
    status = Column(String, default="active")
    
    tasks = relationship("DailyTask", back_populates="goal")

# TABLE 2 - Daily Tasks
class DailyTask(Base):
    __tablename__ = "daily_tasks"
    
    id = Column(Integer, primary_key=True)
    goal_id = Column(Integer, ForeignKey("learning_goals.id"))
    day_number = Column(Integer)
    topic = Column(String)
    subtopic = Column(String)
    instructions = Column(Text)
    resources = Column(Text)
    estimated_hours = Column(Float)
    status = Column(String, default="pending")
    scheduled_date = Column(DateTime)
    
    goal = relationship("LearningGoal", back_populates="tasks")
    progress = relationship("Progress", back_populates="task")

# TABLE 3 - Progress
class Progress(Base):
    __tablename__ = "progress"
    
    id = Column(Integer, primary_key=True)
    task_id = Column(Integer, ForeignKey("daily_tasks.id"))
    completed_at = Column(DateTime, default=datetime.now)
    quiz_score = Column(Float)
    notes = Column(Text)
    adjusted = Column(Boolean, default=False)
    
    task = relationship("DailyTask", back_populates="progress")

# TABLE 4 - Resources
class Resource(Base):
    __tablename__ = "resources"
    
    id = Column(Integer, primary_key=True)
    topic = Column(String)
    title = Column(String)
    url = Column(String)
    type = Column(String)  # video/doc/tutorial/github
    quality_score = Column(Float)

# Create all tables
def init_db():
    Base.metadata.create_all(engine)
    print("✅ Database tables created successfully!")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

if __name__ == "__main__":
    init_db()