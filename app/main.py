from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
import tempfile
import os
import json
from datetime import datetime
from sqlalchemy.orm import Session

from .database import get_db, engine, Base
from .models import Resume, Job, JobMatch
from .resume_parser import ResumeParser
from .job_crawler import JobCrawler
from .job_recommender import JobRecommender

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
resume_parser = ResumeParser()
job_crawler = JobCrawler()
job_recommender = JobRecommender()

@app.post("/upload-resume")
async def upload_resume(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name

    try:
        # Extract text and skills from resume
        text = resume_parser.extract_text_from_pdf(content)
        skills = resume_parser.extract_skills(text)

        # Store resume in vector store
        vector_id = resume_parser.store_resume(text, skills)

        # Store resume in database
        db_resume = Resume(
            content=text,
            skills=skills,
            vector_id=vector_id
        )
        db.add(db_resume)
        db.commit()
        db.refresh(db_resume)

        return {
            "message": "Resume processed successfully",
            "resume_id": db_resume.id,
            "skills": skills
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temporary file
        os.unlink(temp_path)

@app.get("/recommend-jobs/{resume_id}")
async def recommend_jobs(
    resume_id: int,
    min_match: float = 0.3,
    max_jobs: int = 5,
    db: Session = Depends(get_db)
):
    try:
        # Get resume from database
        resume = db.query(Resume).filter(Resume.id == resume_id).first()
        if not resume:
            raise HTTPException(status_code=404, detail="Resume not found")

        # Get job recommendations
        recommendations = job_recommender.recommend_jobs(
            resume.vector_id,
            min_skill_match=min_match,
            max_recommendations=max_jobs
        )

        # Store job matches in database
        for rec in recommendations:
            job_match = JobMatch(
                resume_id=resume_id,
                job_id=rec['job_id'],
                skill_match_score=rec['skill_match_score'],
                semantic_similarity=rec['semantic_similarity'],
                final_score=rec['final_score']
            )
            db.add(job_match)
        db.commit()

        return recommendations

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/crawl-jobs")
async def crawl_jobs(
    query: str,
    location: str = "",
    max_jobs: int = 10,
    db: Session = Depends(get_db)
):
    try:
        # Crawl jobs from source
        jobs = job_crawler.crawl_indeed_jobs(query, location, max_jobs)

        # Store jobs in database and vector store
        stored_jobs = []
        for job in jobs:
            # Store in vector store
            vector_id = job_crawler.store_job(job)

            # Store in database
            db_job = Job(
                title=job['title'],
                company=job['company'],
                description=job['description'],
                skills_required=job_crawler.extract_skills_from_description(job['description']),
                vector_id=vector_id,
                source=job['source'],
                url=job.get('url', '')
            )
            db.add(db_job)
            stored_jobs.append(job)

        db.commit()
        return stored_jobs

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
