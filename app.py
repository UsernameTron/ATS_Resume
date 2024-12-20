import os
import logging
import openai
import nltk
import re
import base64
from collections import Counter
from PyPDF2 import PdfReader
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from io import BytesIO
import streamlit as st
from dotenv import load_dotenv  # Import dotenv

# Load environment variables from the .env file
dotenv_path = '/Users/pconnor/Library/CloudStorage/OneDrive-Personal/Streamlit/ATS_Resume/.env'
load_dotenv(dotenv_path)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set OpenAI and Google API keys from environment variables
openai.api_key = os.getenv('OPENAI_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')  # If used elsewhere in your app

# Download NLTK data quietly
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)


class ResumeAnalyzer:
    def __init__(self, openai_api_key, google_api_key):
        openai.api_key = openai_api_key
        self.google_api_key = google_api_key

    @st.cache_data
    def extract_text_from_pdf(self, pdf_stream):
        """Extract text from a PDF file stream."""
        try:
            reader = PdfReader(pdf_stream)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            logging.info("Successfully extracted text from PDF.")
            return text
        except Exception as e:
            logging.error(f"Error extracting text from PDF: {e}", exc_info=True)
            st.error("An unexpected error occurred while extracting text from the PDF. Please ensure the file is not corrupted and try again.")
            return ""

    def extract_keywords(self, text):
        """Extract significant keywords from the text using NLP techniques."""
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text.lower())
        keywords = [word for word in words if word.isalnum() and word not in stop_words]
        logging.info(f"Extracted {len(keywords)} keywords from text.")
        return keywords

    def calculate_keyword_density(self, text, keywords):
        """
        Calculate the keyword density in the text based on the provided keywords.

        Args:
            text (str): The text in which to calculate keyword density.
            keywords (list): A list of keywords to calculate density for.

        Returns:
            float: The keyword density percentage rounded to two decimal places.
        """
        # Extract all words from the text
        words = re.findall(r'\w+', text.lower())
        total_words = len(words)

        # Count occurrences of each keyword
        keyword_counts = Counter(word for word in words if word in keywords)
        total_keywords = sum(keyword_counts.values())

        # Calculate density
        density = (total_keywords / total_words) * 100 if total_words > 0 else 0
        density_rounded = round(density, 2)

        logging.info(f"Calculated keyword density: {density_rounded}%")
        return density_rounded

    def generate_analysis(self, resume_text, job_description):
        """Generate analysis using OpenAI."""
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
            response = openai.ChatCompletion.create(
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
            st.error("An error occurred while generating the analysis.")
            return "An error occurred while generating the analysis."

    def generate_optimized_resume(self, resume_text, job_description):
        """Generate an optimized résumé using OpenAI."""
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
            response = openai.ChatCompletion.create(
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

    def download_text(self, text, filename):
        """Allow users to download text as a file."""
        try:
            b64 = base64.b64encode(text.encode()).decode()
            href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download {filename}</a>'
            logging.info(f"Generated download link for {filename}.")
            return href
        except Exception as e:
            logging.error(f"Error generating download link: {e}", exc_info=True)
            return ""


# ====================== #
#        Main App        #
# ====================== #

def main():
    # Initialize ResumeAnalyzer
    analyzer = ResumeAnalyzer(openai_api_key=openai.api_key, google_api_key=google_api_key)

    # Title
    st.title("AI-Powered ATS Résumé Analyzer")

    # Instructions
    st.markdown("""
    This application analyzes your résumé against a provided job description to assess the match quality and generates an optimized résumé tailored for Applicant Tracking Systems.

    **Steps to Use:**
    1. Upload your résumé in the section below.
    2. Enter or paste the job description in the text area.
    3. Click on "Generate Analysis" to view the analysis.
    4. Click on "Generate Optimized Résumé" to receive a tailored résumé optimized for ATS.
    """)

    # Sidebar Configuration (Optional)
    st.sidebar.header("Configuration")

    # File Uploader for Résumé
    st.header("Upload Your Résumé")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        # Read the uploaded PDF
        resume_bytes = uploaded_file.read()
        resume_stream = BytesIO(resume_bytes)

        # Extract text from the uploaded PDF
        resume_text = analyzer.extract_text_from_pdf(resume_stream)

        if resume_text:
            st.success("Résumé uploaded and text extracted successfully!")

            # Display the résumé text
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
                    with st.spinner("Generating analysis..."):
                        analysis = analyzer.generate_analysis(resume_text, job_description)
                        if analysis:
                            st.success("Analysis generated successfully!")
                            st.subheader("Analysis")
                            st.write(analysis)

                            # Keyword Extraction and Density Calculation
                            keywords = analyzer.extract_keywords(job_description)
                            density = analyzer.calculate_keyword_density(resume_text, keywords)
                            st.write(f"**Keyword Density:** {density}%")
                        else:
                            st.error("Failed to generate analysis.")

            # Generate Optimized Résumé Button
            if st.button("Generate Optimized Résumé"):
                if not job_description.strip():
                    st.warning("Please enter a job description to proceed.")
                else:
                    with st.spinner("Generating optimized résumé..."):
                        optimized_resume = analyzer.generate_optimized_resume(resume_text, job_description)
                        if optimized_resume:
                            st.success("Optimized résumé generated successfully!")
                            st.subheader("Optimized Résumé")
                            st.text_area("Your Optimized Résumé:", value=optimized_resume, height=600)
                            st.markdown(analyzer.download_text(optimized_resume, "optimized_resume.txt"), unsafe_allow_html=True)
                        else:
                            st.error("Failed to generate optimized résumé.")
        else:
            st.error("Failed to extract text from the résumé. Please check the PDF formatting.")
    else:
        st.warning("Please upload your Résumé to proceed.")

    # Footer
    st.markdown("---")


if __name__ == "__main__":
    main()
