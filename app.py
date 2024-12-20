# app.py

import os
import io
import base64
import logging
import asyncio
from dotenv import load_dotenv
import streamlit as st
import openai
import aiohttp
from PyPDF2 import PdfReader
from collections import Counter
import re
import pandas as pd
import plotly.express as px
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ====================== #
#       Configuration    #
# ====================== #

# Initialize logging
logging.basicConfig(
    filename='app.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Load NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load environment variables
load_dotenv()

# Configure OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Please set your OPENAI_API_KEY in the .env file.")
    st.stop()
openai.api_key = OPENAI_API_KEY

# ====================== #
#      Streamlit Setup   #
# ====================== #

# Set page configuration with custom theme
st.set_page_config(
    page_title="AI-Powered ATS Résumé Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "### **AI-Powered ATS Résumé Analyzer**\nThis app analyzes your résumé against a job description to optimize it for Applicant Tracking Systems."
    }
)

# Apply custom CSS for enhanced styling (optional)
st.markdown("""
    <style>
    /* Custom CSS */
    .stTitle {
        color: #4B8BBE;
        text-align: center;
    }
    .stButton>button {
        background-color: #4B8BBE;
        color: white;
    }
    .stTextArea textarea {
        background-color: #F0F2F6;
        color: #333333;
    }
    </style>
    """, unsafe_allow_html=True)

# ====================== #
#        Functions       #
# ====================== #

@st.cache_data
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        logging.info("Successfully extracted text from PDF.")
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def extract_keywords(text):
    """Extract significant keywords from the text using NLP techniques."""
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    keywords = [word for word in words if word.isalnum() and word not in stop_words]
    logging.info(f"Extracted {len(keywords)} keywords from text.")
    return keywords

def calculate_keyword_density(text, keywords):
    """Calculate the keyword density in the text based on the provided keywords."""
    words = re.findall(r'\w+', text.lower())
    total_words = len(words)
    keyword_counts = Counter(word for word in words if word in keywords)
    total_keywords = sum(keyword_counts.values())
    density = (total_keywords / total_words) * 100 if total_words > 0 else 0
    density_rounded = round(density, 2)
    logging.info(f"Calculated keyword density: {density_rounded}%")
    return density_rounded

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
        response = await openai.Completion.acreate(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=800,
            temperature=0.7,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        analysis = response.choices[0].text.strip()
        logging.info("Successfully generated analysis.")
        return analysis
    except Exception as e:
        logging.error(f"Error generating analysis: {e}")
        st.error(f"Error generating analysis: {e}")
        return "An error occurred while generating the analysis."

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
        response = await openai.Completion.acreate(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=1500,
            temperature=0.7,
            top_p=1,
            frequency_penalty=0.5,
            presence_penalty=0.3
        )
        optimized_resume = response.choices[0].text.strip()
        logging.info("Successfully generated optimized résumé.")
        return optimized_resume
    except Exception as e:
        logging.error(f"Error generating optimized résumé: {e}")
        st.error(f"Error generating optimized résumé: {e}")
        return "An error occurred while generating the optimized résumé."

def download_text(text, filename):
    """Allow users to download text as a file."""
    try:
        b64 = base64.b64encode(text.encode()).decode()
        href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download {filename}</a>'
        logging.info(f"Generated download link for {filename}.")
        return href
    except Exception as e:
        logging.error(f"Error generating download link: {e}")
        return ""

def display_recommendations(recommendations):
    """Display professional recommendations in two columns."""
    col1, col2 = st.columns(2)
    for i, rec in enumerate(recommendations):
        col = col1 if i % 2 == 0 else col2
        with col:
            st.markdown(f"**{rec['name']}**")
            st.write(rec['text'])

# ====================== #
#        Main App        #
# ====================== #

# Title
st.title("AI-Powered ATS Résumé Analyzer")

# Instructions
st.markdown("""
This application analyzes your résumé against a provided job description to assess the match quality and generates an optimized résumé tailored for Applicant Tracking Systems.

**Steps to Use:**
1. Ensure your résumé is placed in the `resumes/` folder in this repository (e.g., `resumes/Resume.pdf`).
2. Enter or paste the job description in the text area below.
3. Click on "Generate Analysis" to view the analysis.
4. Click on "Generate Optimized Résumé" to receive a tailored résumé optimized for ATS.
""")

# Sidebar Configuration (Optional)
st.sidebar.header("Configuration")

# Path to résumé
resume_path = 'resumes/Resume.pdf'

# Check if résumé exists
if not os.path.exists(resume_path):
    st.error(f"Résumé file not found at `{resume_path}`. Please ensure the résumé is placed in the `resumes/` folder.")
    st.stop()

# Extract résumé text
with st.spinner("Extracting résumé text..."):
    resume_text = extract_text_from_pdf(resume_path)
    if not resume_text:
        st.error("Failed to extract text from the résumé. Please check the PDF formatting.")
        st.stop()
    st.success("Résumé text extracted successfully!")

# Display résumé (Optional)
with st.expander("View Résumé"):
    st.write(resume_text)

# Input for Job Description
st.header("Enter Job Description")
job_description = st.text_area("Paste the Job Description here:", height=300)

# Generate Analysis Button
if st.button("Generate Analysis"):
    if not job_description.strip():
        st.warning("Please enter a job description to proceed.")
    else:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # For Windows compatibility
        with st.spinner("Generating analysis..."):
            analysis = asyncio.run(async_generate_analysis(resume_text, job_description))
            if analysis:
                st.success("Analysis generated successfully!")
                st.subheader("Analysis")
                st.write(analysis)
                # Enhanced Keyword Extraction
                keywords = extract_keywords(job_description)
                density = calculate_keyword_density(resume_text, keywords)
                st.write(f"**Keyword Density:** {density}%")
            else:
                st.error("Failed to generate analysis.")

# Generate Optimized Résumé Button
if st.button("Generate Optimized Résumé"):
    if not job_description.strip():
        st.warning("Please enter a job description to proceed.")
    else:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # For Windows compatibility
        with st.spinner("Generating optimized résumé..."):
            optimized_resume = asyncio.run(async_generate_optimized_resume(resume_text, job_description))
            if optimized_resume:
                st.success("Optimized résumé generated successfully!")
                st.subheader("Optimized Résumé")
                st.text_area("Your Optimized Résumé:", value=optimized_resume, height=600)
                st.markdown(download_text(optimized_resume, "optimized_resume.txt"), unsafe_allow_html=True)
            else:
                st.error("Failed to generate optimized résumé.")

# Footer
st.markdown("---")

# ====================== #
#      Additional Info    #
# ====================== #

# Example recommendations data (optional)
recommendations = [
    {"name": "John Doe", "text": "An outstanding professional!"},
    {"name": "Jane Smith", "text": "Brings creativity and insight to every project."},
    {"name": "Alex Johnson", "text": "A reliable team player who exceeds expectations."},
    {"name": "Maria Garcia", "text": "Combines technical skill with compelling storytelling."}
]

st.subheader("Professional Recommendations")
display_recommendations(recommendations)

# ====================== #
#        End of App       #
# ====================== #