# AI Document Structurer (Streamlit App)

This Streamlit application extracts structured information from unorganized PDF documents and converts it into a clean **Key | Value | Comments** Excel file. It uses paragraph normalization and rule-based text extraction to preserve the original context while organizing the data.

## Features

- Upload any PDF directly in the browser
- Automatic text extraction using PyPDF2
- Paragraph normalization to fix broken line wraps
- Extraction of:
  - Personal details (name, date of birth, age)
  - Employment history
  - Education details
  - Skills and ratings
  - Fallback text lines
- Includes original matched context in a **Comments** column
- Instant Excel download inside the app

## Installation (Local)

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
streamlit run app_streamlit.py
