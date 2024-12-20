import os
import logging
import asyncio
import openai
import nltk
import requests
from PyPDF2 import PdfReader
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from io import BytesIO
import streamlit as st
from dotenv import load_dotenv
from collections import Counter
import re
import base64
from html import escape
from urllib.parse import urljoin

# Load environment variables from the .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set OpenAI API key from environment variables
openai.api_key = os.getenv('OPENAI_API_KEY')

# GitHub Repository Configuration
GITHUB_USER = "UsernameTron"  # Replace with your GitHub username
GITHUB_REPO = "ATS_Resume"     # Replace with your repository name
GITHUB_BRANCH = "main"         # Replace with your repository's default branch
RESUME_FOLDER = ""              # Root directory in the repo where resumes are stored

# Base URL for GitHub API
GITHUB_API_BASE = "https://api.github.com/"

# Download NLTK data quietly
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

class ResumeAnalyzer:
    @staticmethod
    @st.cache_data
    def list_resumes():
        """
        List all PDF resumes in the specified GitHub repository folder.
        """
        api_url = urljoin(GITHUB_API_BASE, f"repos/{GITHUB_USER}/{GITHUB_REPO}/contents/{RESUME_FOLDER}")
        response = requests.get(api_url)
        if response.status_code == 200:
            files = response.json()
            pdf_files = [file for file in files if file['name'].lower().endswith('.pdf')]
            return pdf_files
        else:
            logging.error(f"Failed to fetch resumes from GitHub: {response.status_code}")
            st.error("Failed to fetch resumes from GitHub. Please check the repository settings.")
            return []

    @staticmethod
    @st.cache_data
    def fetch_resume_content(download_url):
        """
        Fetch the resume PDF content from GitHub.
        """
        try:
            response = requests.get(download_url)
            if response.status_code == 200:
                return BytesIO(response.content)
            else:
                logging.error(f"Failed to download resume: {response.status_code}")
                st.error("Failed to download the selected resume.")
                return None
        except Exception as e:
            logging.error(f"Exception while downloading resume: {e}", exc_info=True)
            st.error("An error occurred while downloading the resume.")
            return None

    @staticmethod
    @st.cache_data
    def extract_text_from_pdf(pdf_stream):
        """Extract text from a PDF file stream."""
        try:
            logging.debug("Starting PDF text extraction.")
            reader = PdfReader(pdf_stream)
            text = ""
            for page_number, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                    logging.debug(f"Extracted text from page {page_number}.")
                else:
                    logging.warning(f"No text found on page {page_number}.")
            logging.info("Successfully extracted text from PDF.")
            return text
        except Exception as e:
            logging.error(f"Error extracting text from PDF: {e}", exc_info=True)
            st.error("An unexpected error occurred while extracting text from the PDF. Please ensure the file is not corrupted and try again.")
            return ""

    @staticmethod
    def extract_keywords(text):
        """Extract significant keywords from the text using NLP techniques."""
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text.lower())
        keywords = [word for word in words if word.isalnum() and word not in stop_words]
        logging.info(f"Extracted {len(keywords)} keywords from text.")
        return keywords

    @staticmethod
    def calculate_keyword_density(text, keywords):
        """
        Calculate the keyword density in the text based on the provided keywords.

        Args:
            text (str): The text in which to calculate keyword density.
            keywords (list): A list of keywords to calculate density for.

        Returns:
            float: The keyword density percentage rounded to two decimal places.
        """
        words = re.findall(r'\w+', text.lower())
        total_words = len(words)
        keyword_counts = Counter(word for word in words if word in keywords)
        total_keywords = sum(keyword_counts.values())
        density = (total_keywords / total_words) * 100 if total_words > 0 else 0
        density_rounded = round(density, 2)
        logging.info(f"Calculated keyword density: {density_rounded}%")
        return density_rounded

    @staticmethod
    async def async_generate_analysis(resume_text, job_description):
        """Asynchronously generate analysis using OpenAI."""
        try:
            prompt = f"""
            You are an ATS expert. Compare the following résumé to the job description.

            Résumé:
            {resume_text}

            Job Description:
            {job_description}

            Provide a detailed analysis that includes:
            1. Percentage match between the résumé and the job description.
            2. Missing keywords or skills in the résumé.
            3. Strengths of the candidate based on the résumé.
            4. Areas for improvement in the résumé to better match the job description.
            """
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.7
            )
            analysis = response.choices[0].message['content'].strip()
            logging.info("Successfully generated analysis.")
            return analysis
        except Exception as e:
            logging.error(f"Error generating analysis: {e}", exc_info=True)
            st.error("An unexpected error occurred while generating the analysis.")
            return "An error occurred while generating the analysis."

    @staticmethod
    async def async_generate_optimized_resume(resume_text, job_description):
        """Asynchronously generate an optimized résumé using OpenAI."""
        try:
            prompt = f"""
            You are an expert resume writer specializing in optimizing résumés for ATS (Applicant Tracking Systems).
            Your task is to rewrite the following résumé to achieve a 90% ATS match and ensure a minimum of 4.8% keyword density based on the provided job description.

            Job Description:
            {job_description}

            Résumé:
            {resume_text}

            Optimized Résumé:
            """
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.7
            )
            optimized_resume = response.choices[0].message['content'].strip()
            logging.info("Successfully generated optimized résumé.")
            return optimized_resume
        except Exception as e:
            logging.error(f"Error generating optimized résumé: {e}", exc_info=True)
            st.error("An error occurred while generating the optimized résumé.")
            return "An error occurred while generating the optimized résumé."

    @staticmethod
    def download_text(text, filename):
        """Allow users to download text as a file."""
        try:
            b64 = base64.b64encode(text.encode()).decode()
            href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download {filename}</a>'
            logging.info(f"Generated download link for {filename}.")
            return href
        except Exception as e:
            logging.error(f"Error generating download link: {e}", exc_info=True)
            return ""

    @staticmethod
    def sanitize_input(user_input):
        """Sanitize user input to prevent security vulnerabilities."""
        return escape(user_input)


def main():
    # Title
    st.title("AI-Powered ATS Résumé Analyzer")

    # Instructions
    st.markdown("""
    This application analyzes your résumé against a provided job description to assess the match quality and generates an optimized résumé tailored for Applicant Tracking Systems.

    **Steps to Use:**
    1. Select your résumé from the GitHub repository.
    2. Enter or paste the job description in the text area.
    3. Click on "Generate Analysis" to view the analysis.
    4. Click on "Generate Optimized Résumé" to receive a tailored résumé optimized for ATS.
    """)

    # Sidebar Configuration (Optional)
    st.sidebar.header("Configuration")
    st.sidebar.markdown("Configure any additional settings here.")

    # List Resumes from GitHub
    st.header("Select Your Résumé")
    resumes = ResumeAnalyzer.list_resumes()
    if resumes:
        resume_names = [resume['name'] for resume in resumes]
        selected_resume = st.selectbox("Choose a résumé to analyze:", resume_names)

        # Get the selected resume's download URL
        resume_file = next((file for file in resumes if file['name'] == selected_resume), None)
        if resume_file:
            download_url = resume_file['download_url']
            resume_stream = ResumeAnalyzer.fetch_resume_content(download_url)

            if resume_stream:
                # Extract text from the fetched PDF
                resume_text = ResumeAnalyzer.extract_text_from_pdf(resume_stream)

                if resume_text:
                    st.success("Résumé text extracted successfully!")

                    # Display the résumé text
                    with st.expander("View Résumé"):
                        st.write(resume_text)

                    # Input for Job Description
                    st.header("Enter Job Description")
                    job_description = ResumeAnalyzer.sanitize_input(
                        st.text_area("Paste the Job Description here:", height=300)
                    )

                    # Generate Analysis Button
                    if st.button("Generate Analysis"):
                        if not job_description.strip():
                            st.warning("Please enter a job description to proceed.")
                        else:
                            with st.spinner("Generating analysis..."):
                                try:
                                    analysis = asyncio.run(ResumeAnalyzer.async_generate_analysis(resume_text, job_description))
                                    if analysis:
                                        st.success("Analysis generated successfully!")
                                        st.subheader("Analysis")
                                        st.write(analysis)

                                        # Keyword Extraction and Density Calculation
                                        keywords = ResumeAnalyzer.extract_keywords(job_description)
                                        density = ResumeAnalyzer.calculate_keyword_density(resume_text, keywords)
                                        st.write(f"**Keyword Density:** {density}%")
                                    else:
                                        st.error("Failed to generate analysis.")
                                except Exception as e:
                                    st.error(f"An unexpected error occurred: {e}")

                    # Generate Optimized Résumé Button
                    if st.button("Generate Optimized Résumé"):
                        if not job_description.strip():
                            st.warning("Please enter a job description to proceed.")
                        else:
                            with st.spinner("Generating optimized résumé..."):
                                try:
                                    optimized_resume = asyncio.run(ResumeAnalyzer.async_generate_optimized_resume(resume_text, job_description))
                                    if optimized_resume:
                                        st.success("Optimized résumé generated successfully!")
                                        st.subheader("Optimized Résumé")
                                        st.text_area("Your Optimized Résumé:", value=optimized_resume, height=600)
                                        st.markdown(ResumeAnalyzer.download_text(optimized_resume, "optimized_resume.txt"), unsafe_allow_html=True)
                                    else:
                                        st.error("Failed to generate optimized résumé.")
                                except Exception as e:
                                    st.error(f"An unexpected error occurred: {e}")
                else:
                    st.error("Failed to extract text from the résumé. Please check the PDF formatting.")
            else:
                st.error("Failed to download the selected résumé.")
    else:
        st.error("No resumes found in the GitHub repository. Please ensure resumes are uploaded to the specified folder.")

    # Footer
    st.markdown("---")
    st.markdown("© 2024 AI-Powered ATS Résumé Analyzer. All rights reserved.")


if __name__ == "__main__":
    main()
