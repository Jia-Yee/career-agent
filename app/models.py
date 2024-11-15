from sqlalchemy import Column, Integer, String, DateTime, JSON, Float, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base

class Resume(Base):
    __tablename__ = "resumes"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(String)
    skills = Column(JSON)
    vector_id = Column(String, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    company = Column(String)
    description = Column(String)
    skills_required = Column(JSON)
    vector_id = Column(String, unique=True)
    source = Column(String)
    url = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class JobMatch(Base):
    __tablename__ = "job_matches"

    id = Column(Integer, primary_key=True, index=True)
    resume_id = Column(Integer, ForeignKey("resumes.id"))
    job_id = Column(Integer, ForeignKey("jobs.id"))
    skill_match_score = Column(Float)
    semantic_similarity = Column(Float)
    final_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    resume = relationship("Resume")
    job = relationship("Job")
