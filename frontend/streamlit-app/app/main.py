import streamlit as st
import requests
from pathlib import Path
import json
import time

# Backend API URL
API_URL = "https://app-soplrpfo.fly.dev"

def display_skills(skills):
    """Display extracted skills in an organized manner"""
    st.subheader("📊 Skills Analysis")

    for category, skill_list in skills.items():
        if skill_list:
            with st.expander(f"{category.replace('_', ' ').title()}"):
                for skill in skill_list:
                    st.write(f"- {skill}")

def display_jobs(jobs):
    """Display recommended jobs with detailed information"""
    st.subheader("🎯 Recommended Jobs")

    for job in jobs:
        score_color = "green" if job["final_score"] >= 0.7 else "orange" if job["final_score"] >= 0.5 else "red"

        with st.expander(
            f"{job['title']} at {job['company']} - Match: :{score_color}[{job['final_score']:.0%}]"
        ):
            st.write("**Company:**", job["company"])
            st.write("**Source:**", job["source"])
            st.write("**Description:**", job["description"])

            # Display match scores
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Skill Match", f"{job['skill_match_score']:.0%}")
            with col2:
                st.metric("Semantic Match", f"{job['semantic_similarity']:.0%}")

            # Display required skills
            if job.get("skills_required"):
                st.write("**Required Skills:**")
                for category, skills in job["skills_required"].items():
                    if skills:
                        st.write(f"*{category.replace('_', ' ').title()}:*")
                        st.write(", ".join(skills))

def main():
    st.set_page_config(
        page_title="Career Agent",
        page_icon="🎯",
        layout="wide"
    )

    st.title("🎯 Career Agent - Your Personal Career Assistant")
    st.write("""
    Upload your resume to get personalized job recommendations based on your skills and experience.
    Our AI-powered system will analyze your resume and match you with the most relevant job opportunities.
    """)

    # File upload section
    uploaded_file = st.file_uploader(
        "📄 Upload your resume (PDF format)",
        type=['pdf'],
        help="Upload your resume in PDF format to get started"
    )

    if uploaded_file:
        with st.spinner("🔍 Analyzing your resume..."):
            try:
                # Prepare the file for upload
                files = {"file": ("resume.pdf", uploaded_file.getvalue(), "application/pdf")}

                # Upload resume and get initial analysis
                response = requests.post(f"{API_URL}/upload-resume", files=files)

                if response.status_code == 200:
                    data = response.json()
                    resume_id = data.get("resume_id")

                    # Create tabs for different sections
                    tab1, tab2 = st.tabs(["📊 Skills Analysis", "💼 Job Matches"])

                    with tab1:
                        # Display extracted skills
                        if "skills" in data:
                            display_skills(data["skills"])
                        else:
                            st.warning("No skills were extracted from the resume")

                    with tab2:
                        # Get and display job recommendations
                        if resume_id:
                            with st.spinner("🔍 Finding matching jobs..."):
                                jobs_response = requests.get(
                                    f"{API_URL}/recommend-jobs/{resume_id}"
                                )

                                if jobs_response.status_code == 200:
                                    jobs_data = jobs_response.json()
                                    if jobs_data:
                                        display_jobs(jobs_data)
                                    else:
                                        st.info("No matching jobs found")
                                else:
                                    st.error("Failed to fetch job recommendations")
                        else:
                            st.error("Failed to process resume")

                else:
                    st.error("Error processing resume. Please try again.")
                    st.write("Error details:", response.text)

            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to the server. Please try again later. {str(e)}")

    # Additional information section
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        The Career Agent helps you find the perfect job match by:
        1. Analyzing your resume
        2. Extracting relevant skills
        3. Matching with job opportunities
        4. Providing personalized recommendations
        """)

        st.header("🔑 Key Features")
        st.write("""
        - AI-powered resume analysis
        - Comprehensive skill extraction
        - Real-time job matching
        - Detailed match scoring
        - Skill gap analysis
        """)

if __name__ == "__main__":
    main()
