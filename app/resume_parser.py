from typing import List, Dict
from pathlib import Path
from datetime import datetime
import re
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import json
import os

class ResumeParser:
    def __init__(self):
        # Initialize embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

        # Initialize vector store
        self.vector_store = Chroma(
            persist_directory="./data/chroma",
            embedding_function=self.embeddings
        )

        # Load skills database
        self.skills_db = self._load_skills_db()

        # Initialize text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    def _load_skills_db(self) -> Dict[str, List[str]]:
        """Load or create skills database with categories"""
        skills_db = {
            "programming_languages": [
                "python", "java", "javascript", "c++", "ruby", "php", "swift",
                "kotlin", "go", "rust", "typescript", "scala", "r", "matlab"
            ],
            "frameworks": [
                "react", "angular", "vue", "django", "flask", "spring",
                "express", "fastapi", "pytorch", "tensorflow", "keras"
            ],
            "databases": [
                "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
                "cassandra", "oracle", "sql server", "sqlite"
            ],
            "cloud_platforms": [
                "aws", "azure", "google cloud", "heroku", "digitalocean",
                "kubernetes", "docker", "terraform"
            ],
            "tools": [
                "git", "jenkins", "jira", "confluence", "slack", "postman",
                "webpack", "npm", "yarn", "maven", "gradle"
            ]
        }
        return skills_db

    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extract text content from PDF bytes"""
        try:
            pdf = PdfReader(pdf_content)
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")

    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        """Extract skills from text using categorized pattern matching"""
        text = text.lower()
        found_skills = {category: [] for category in self.skills_db.keys()}

        for category, skills in self.skills_db.items():
            for skill in skills:
                # Use word boundaries for exact matches
                pattern = r'\b' + re.escape(skill) + r'\b'
                if re.search(pattern, text):
                    found_skills[category].append(skill)

        return found_skills

    def store_resume(self, text: str, skills: Dict[str, List[str]]) -> str:
        """Store resume text and skills in vector store"""
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)

        # Create metadata with extracted skills
        metadata = {
            "skills": json.dumps(skills),
            "timestamp": str(datetime.now())
        }

        # Store chunks with metadata
        try:
            ids = self.vector_store.add_texts(
                texts=chunks,
                metadatas=[metadata] * len(chunks)
            )
            return ids[0]  # Return first chunk ID as resume ID
        except Exception as e:
            raise Exception(f"Error storing resume in vector store: {str(e)}")

    def search_similar_resumes(self, query_text: str, n_results: int = 5) -> List[Dict]:
        """Search for similar resumes based on query text"""
        try:
            results = self.vector_store.similarity_search_with_score(
                query_text,
                k=n_results
            )
            return [
                {
                    "text": result[0].page_content,
                    "metadata": result[0].metadata,
                    "score": result[1]
                }
                for result in results
            ]
        except Exception as e:
            raise Exception(f"Error searching similar resumes: {str(e)}")

    def get_skills_statistics(self) -> Dict[str, Dict[str, int]]:
        """Get statistics about stored skills"""
        try:
            all_docs = self.vector_store.get()
            skills_count = {category: {} for category in self.skills_db.keys()}

            for doc in all_docs:
                if "skills" in doc.metadata:
                    skills = json.loads(doc.metadata["skills"])
                    for category, skill_list in skills.items():
                        for skill in skill_list:
                            if skill not in skills_count[category]:
                                skills_count[category][skill] = 0
                            skills_count[category][skill] += 1

            return skills_count
        except Exception as e:
            raise Exception(f"Error getting skills statistics: {str(e)}")
