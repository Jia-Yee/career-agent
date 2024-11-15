from typing import List, Dict, Optional
import json
from datetime import datetime
import numpy as np
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

class JobRecommender:
    def __init__(self):
        # Initialize embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

        # Initialize vector stores
        self.resume_store = Chroma(
            persist_directory="./data/chroma",
            embedding_function=self.embeddings
        )
        self.job_store = Chroma(
            persist_directory="./data/jobs_chroma",
            embedding_function=self.embeddings
        )

    def calculate_skill_match_score(self, resume_skills: Dict[str, List[str]],
                                  job_skills: Dict[str, List[str]]) -> float:
        """Calculate skill match score between resume and job"""
        total_weight = 0
        matched_weight = 0

        # Category weights for scoring
        weights = {
            'programming_languages': 0.3,
            'frameworks': 0.25,
            'databases': 0.2,
            'cloud_platforms': 0.15,
            'tools': 0.1
        }

        for category in weights:
            if category in resume_skills and category in job_skills:
                resume_skill_set = set(resume_skills[category])
                job_skill_set = set(job_skills[category])

                if job_skill_set:  # Only consider categories with required skills
                    weight = weights[category]
                    total_weight += weight

                    # Calculate matched skills
                    matched_skills = resume_skill_set.intersection(job_skill_set)
                    if job_skill_set:
                        category_score = len(matched_skills) / len(job_skill_set)
                        matched_weight += weight * category_score

        # Return normalized score
        return matched_weight / total_weight if total_weight > 0 else 0.0

    def get_semantic_similarity_score(self, resume_text: str, job_description: str) -> float:
        """Calculate semantic similarity between resume and job description"""
        try:
            # Get embeddings for resume and job description
            resume_embedding = self.embeddings.embed_documents([resume_text])[0]
            job_embedding = self.embeddings.embed_documents([job_description])[0]

            # Calculate cosine similarity
            similarity = np.dot(resume_embedding, job_embedding) / \
                        (np.linalg.norm(resume_embedding) * np.linalg.norm(job_embedding))

            return float(similarity)
        except Exception as e:
            print(f"Error calculating semantic similarity: {str(e)}")
            return 0.0

    def recommend_jobs(self, resume_id: str,
                      min_skill_match: float = 0.3,
                      max_recommendations: int = 5) -> List[Dict]:
        """Recommend jobs based on resume"""
        try:
            # Get resume data
            resume_docs = self.resume_store.get([resume_id])
            if not resume_docs:
                raise ValueError(f"Resume with ID {resume_id} not found")

            resume_doc = resume_docs[0]
            resume_skills = json.loads(resume_doc.metadata.get('skills', '{}'))
            resume_text = resume_doc.page_content

            # Get all jobs
            all_jobs = self.job_store.get()

            recommendations = []
            for job in all_jobs:
                job_skills = json.loads(job.metadata.get('skills', '{}'))

                # Calculate skill match score
                skill_score = self.calculate_skill_match_score(resume_skills, job_skills)

                # Only process jobs with minimum skill match
                if skill_score >= min_skill_match:
                    # Calculate semantic similarity
                    semantic_score = self.get_semantic_similarity_score(
                        resume_text, job.page_content
                    )

                    # Calculate final score (weighted average)
                    final_score = (skill_score * 0.7) + (semantic_score * 0.3)

                    recommendations.append({
                        'job_id': job.id,
                        'title': job.metadata.get('title'),
                        'company': job.metadata.get('company'),
                        'description': job.page_content,
                        'skills_required': job_skills,
                        'skill_match_score': skill_score,
                        'semantic_similarity': semantic_score,
                        'final_score': final_score,
                        'source': job.metadata.get('source'),
                        'crawled_at': job.metadata.get('crawled_at')
                    })

            # Sort by final score and return top recommendations
            recommendations.sort(key=lambda x: x['final_score'], reverse=True)
            return recommendations[:max_recommendations]

        except Exception as e:
            raise Exception(f"Error generating job recommendations: {str(e)}")


    def get_recommendation_insights(self, recommendation: Dict) -> Dict:
        """Generate insights for a job recommendation"""
        try:
            insights = {
                'match_summary': {
                    'overall_score': recommendation['final_score'],
                    'skill_match': recommendation['skill_match_score'],
                    'semantic_match': recommendation['semantic_similarity']
                },
                'skill_breakdown': {},
                'improvement_areas': [],
                'key_matches': []
            }

            # Analyze skill matches and gaps
            for category, skills in recommendation['skills_required'].items():
                if skills:
                    insights['skill_breakdown'][category] = {
                        'required': len(skills),
                        'matched': 0  # This would be calculated based on resume skills
                    }

                    # Add missing critical skills to improvement areas
                    if insights['skill_breakdown'][category]['matched'] < len(skills):
                        insights['improvement_areas'].append(
                            f"Consider developing skills in {category}: {', '.join(skills)}"
                        )

            return insights

        except Exception as e:
            raise Exception(f"Error generating recommendation insights: {str(e)}")
