from typing import List, Dict, Optional
import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
import time
import re
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

class JobCrawler:
    def __init__(self):
        # Initialize embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

        # Initialize vector store for jobs
        self.vector_store = Chroma(
            persist_directory="./data/jobs_chroma",
            embedding_function=self.embeddings
        )

        # Headers for requests
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def crawl_indeed_jobs(self, query: str, location: str = "", max_jobs: int = 10) -> List[Dict]:
        """Crawl job listings from Indeed"""
        base_url = "https://www.indeed.com/jobs"
        jobs = []

        try:
            params = {
                'q': query,
                'l': location,
                'sort': 'date'
            }

            response = requests.get(base_url, params=params, headers=self.headers)
            soup = BeautifulSoup(response.text, 'html.parser')

            job_cards = soup.find_all('div', class_='job_seen_beacon')

            for card in job_cards[:max_jobs]:
                job = self._parse_indeed_job_card(card)
                if job:
                    jobs.append(job)

            return jobs
        except Exception as e:
            print(f"Error crawling Indeed: {str(e)}")
            return []

    def _parse_indeed_job_card(self, card) -> Optional[Dict]:
        """Parse individual Indeed job card"""
        try:
            title = card.find('h2', class_='jobTitle').get_text(strip=True)
            company = card.find('span', class_='companyName').get_text(strip=True)
            description = card.find('div', class_='job-snippet').get_text(strip=True)

            return {
                'title': title,
                'company': company,
                'description': description,
                'source': 'indeed',
                'crawled_at': datetime.now().isoformat()
            }
        except Exception:
            return None

    def extract_skills_from_description(self, description: str) -> Dict[str, List[str]]:
        """Extract skills from job description"""
        # Common skills patterns
        skill_patterns = {
            'programming_languages': r'\b(python|java|javascript|c\+\+|ruby|php|swift|kotlin|go|rust|typescript|scala|r|matlab)\b',
            'frameworks': r'\b(react|angular|vue|django|flask|spring|express|fastapi|pytorch|tensorflow|keras)\b',
            'databases': r'\b(mysql|postgresql|mongodb|redis|elasticsearch|cassandra|oracle|sql server|sqlite)\b',
            'cloud_platforms': r'\b(aws|azure|google cloud|heroku|digitalocean|kubernetes|docker|terraform)\b',
            'tools': r'\b(git|jenkins|jira|confluence|slack|postman|webpack|npm|yarn|maven|gradle)\b'
        }

        found_skills = {category: [] for category in skill_patterns.keys()}
        description = description.lower()

        for category, pattern in skill_patterns.items():
            matches = re.findall(pattern, description)
            if matches:
                found_skills[category] = list(set(matches))

        return found_skills

    def store_job(self, job: Dict) -> str:
        """Store job in vector store"""
        # Extract skills from description
        skills = self.extract_skills_from_description(job['description'])

        # Create metadata
        metadata = {
            'title': job['title'],
            'company': job['company'],
            'source': job['source'],
            'skills': json.dumps(skills),
            'crawled_at': job['crawled_at']
        }

        # Store in vector store
        try:
            ids = self.vector_store.add_texts(
                texts=[job['description']],
                metadatas=[metadata]
            )
            return ids[0]
        except Exception as e:
            raise Exception(f"Error storing job in vector store: {str(e)}")

    def search_similar_jobs(self, query_text: str, n_results: int = 5) -> List[Dict]:
        """Search for similar jobs based on query text"""
        try:
            results = self.vector_store.similarity_search_with_score(
                query_text,
                k=n_results
            )
            return [
                {
                    'description': result[0].page_content,
                    'metadata': result[0].metadata,
                    'score': result[1]
                }
                for result in results
            ]
        except Exception as e:
            raise Exception(f"Error searching similar jobs: {str(e)}")

    def get_job_skills_statistics(self) -> Dict[str, Dict[str, int]]:
        """Get statistics about skills in stored jobs"""
        try:
            all_docs = self.vector_store.get()
            skills_count = {}

            for doc in all_docs:
                if 'skills' in doc.metadata:
                    skills = json.loads(doc.metadata['skills'])
                    for category, skill_list in skills.items():
                        if category not in skills_count:
                            skills_count[category] = {}
                        for skill in skill_list:
                            if skill not in skills_count[category]:
                                skills_count[category][skill] = 0
                            skills_count[category][skill] += 1

            return skills_count
        except Exception as e:
            raise Exception(f"Error getting job skills statistics: {str(e)}")
