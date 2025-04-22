import io
import os
import json
import streamlit as st
import pandas as pd
import pdfplumber
import docx
import openai

from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

#─── ENV & CLIENT SETUP ─────────────────────────────────────────────────────────
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

mongo_uri = os.getenv("MONGO_URI")
client    = MongoClient(mongo_uri)
db        = client["aicruit"]
resumes   = db["resumes"]
job_descriptions = db["job_descriptions"]

#─── RAW TEXT EXTRACTION ────────────────────────────────────────────────────────
def extract_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()
    if name.endswith(".pdf"):
        text = ""
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text
    elif name.endswith(".docx"):
        doc = docx.Document(io.BytesIO(data))
        return "\n".join([p.text for p in doc.paragraphs])
    else:
        st.error("Unsupported file type")
        return ""

# New function to normalize parsed data for MongoDB schema compatibility
def normalize_for_mongodb(parsed_data):
    """Convert arrays and objects to strings for MongoDB schema compatibility"""
    normalized = {}
    
    for key, value in parsed_data.items():
        if isinstance(value, list):
            # Convert list to a formatted string
            if all(isinstance(item, dict) for item in value):
                # Handle list of objects (like work experience entries)
                formatted_items = []
                for item in value:
                    item_str = "\n".join([f"{k}: {v}" for k, v in item.items()])
                    formatted_items.append(item_str)
                normalized[key] = "\n\n".join(formatted_items)
            else:
                # Handle list of strings (like skills)
                normalized[key] = ", ".join(str(item) for item in value)
        elif isinstance(value, dict):
            # Convert dict to a formatted string
            normalized[key] = "\n".join([f"{k}: {v}" for k, v in value.items()])
        else:
            # Keep strings as they are
            normalized[key] = str(value)
    
    return normalized

#─── STRUCTURED PARSING VIA GPT ─────────────────────────────────────────────────
def parse_with_gpt(resume_text: str) -> dict:
    system = {
        "role": "system",
        "content": (
            "You are a JSON-output resume parser. "
            "Given a resume text, extract and return JSON with keys: "
            "work_experience, education, skills, achievements."
        )
    }
    user = {"role": "user", "content": resume_text}

    resp = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[system, user],
        temperature=0
    )
    return json.loads(resp.choices[0].message.content.strip())


# ─── STRUCTURED JOB DESCRIPTION PARSING VIA GPT ────────────────────────────────
def parse_jd_with_gpt(jd_text: str) -> dict:
    """
    Extract structured fields like responsibilities, required_skills, and qualifications
    from a job description using GPT.
    """
    system = {
        "role": "system",
        "content": (
            "You are a job description parser. Extract and return a JSON with these fields:\n"
            "- responsibilities: list of key responsibilities\n"
            "- required_skills: list of skills or technologies required\n"
            "- qualifications: list of qualifications or degrees expected\n"
            "Return valid JSON only."
        )
    }
    user = {"role": "user", "content": jd_text}

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[system, user],
        temperature=0
    )
    
    return json.loads(response.choices[0].message.content.strip())



#─── STREAMLIT APP ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="AIcruit", page_icon=":guardsman:", layout="wide")

# Top Header
col1, col2 = st.columns([0.1, 0.9])
with col1:
    st.image("AIcruit_logo.png", width=150)
with col2:
    st.title("AIcruit")
st.markdown("---")

# Tabs
#tab1, tab2 = st.tabs(["📄 Resume Upload", "📊 Top Resume Matching"])
tab1, tab2, tab3 = st.tabs(["📄 Resume Upload", "📄 Job Description Upload", "📊 Top Resume Matching"])


# Tab 1: Upload & Parse
with tab1:
    st.header("Upload Resume")
    st.write("Upload a resume (PDF or DOCX) to extract key sections and save to MongoDB.")

    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx"])
    if uploaded_file:
        # 1) Extract raw text
        with st.spinner("Extracting text…"):
            raw_text = extract_text(uploaded_file)

        # 2) Parse via GPT-4
        with st.spinner("Parsing sections via GPT-4…"):
            try:
                parsed_data = parse_with_gpt(raw_text)
                # Show the original parsed data
                st.subheader("Original Parsed Data")
                st.json(parsed_data)
                
                # 3) Normalize data for MongoDB schema compatibility
                normalized_parsed = normalize_for_mongodb(parsed_data)
                
                st.subheader("Normalized Data for MongoDB")
                st.json(normalized_parsed)
            except Exception as e:
                st.error(f"Parsing error: {e}")
                parsed_data = None
                normalized_parsed = None

        if normalized_parsed:
            # 4) Insert into MongoDB
            doc = {
                "filename": uploaded_file.name,
                "upload_time": datetime.utcnow(),
                "file_type": uploaded_file.type,
                "file_data": uploaded_file.getvalue(),
                "parsed": normalized_parsed
            }
            
            try:
                result = resumes.insert_one(doc)
                st.success(f"Resume parsed and saved to MongoDB with ID: {result.inserted_id}")
            except Exception as e:
                st.error(f"Error saving to MongoDB: {str(e)}")


# === TAB 2: Job Description Upload (Manual Input + Upload) ===
with tab2:
    st.header("Upload or Enter Job Description")
    st.write("Enter the job title and description manually, or upload a JD file.")

    job_title = st.text_input("Job Title", placeholder="e.g., Software Engineer")

    jd_file = st.file_uploader("Or Upload a Job Description File", type=["pdf", "docx"], key="jd")

    jd_text = ""
    if jd_file:
        with st.spinner("Extracting text from JD file..."):
            jd_text = extract_text(jd_file)

    job_desc_input = st.text_area("Job Description", value=jd_text, height=300)

    if st.button("Parse and Save Job Description"):
        if not job_title.strip():
            st.warning("⚠️ Please provide a job title.")
        elif not job_desc_input.strip():
            st.warning("⚠️ Please enter or upload a job description.")
        else:
            try:
                with st.spinner("Parsing job description via GPT-4..."):
                    parsed_jd = parse_jd_with_gpt(job_desc_input.strip())

                st.success("✅ Parsed JD Structure")
                st.json(parsed_jd)

                jd_doc = {
                    "job_title": job_title.strip(),
                    "upload_time": datetime.utcnow(),
                    "file_type": jd_file.type if jd_file else "text",
                    "filename": jd_file.name if jd_file else "manual_entry",
                    "file_data": jd_file.getvalue() if jd_file else None,
                    "description": job_desc_input.strip(),
                    "parsed": parsed_jd
                }

                result = job_descriptions.insert_one(jd_doc)
                st.success(f"✅ Job Description saved with ID: {result.inserted_id}")

            except Exception as e:
                st.error(f"❌ GPT or MongoDB Error: {e}")


# Tab 3: Matching 
# Tab 3: Matching 
with tab3:
    st.header("Job Description Matching")
    st.write("Select a job description and view top matching resumes.")

    # Fetch unique job titles from MongoDB
    try:
        job_titles = job_descriptions.distinct("job_title")
        if not job_titles:
            st.warning("No job descriptions found. Please upload some in Tab 2.")
        else:
            selected_job_title = st.selectbox("Select Job Description Title", sorted(job_titles))

            if st.button("Find Top Resumes"):
                st.info(f"Finding top resumes for: {selected_job_title}...")
                data = {
                    "Candidate": ["John Doe", "Jane Smith", "Alice Johnson", "Bob Brown"],
                    "Score": [95, 88, 92, 85],
                    "Match %": [90, 80, 85, 75]
                }
                df = pd.DataFrame(data)
                st.subheader("Top Matching Resumes:")
                st.dataframe(df)
    except Exception as e:
        st.error(f"Error loading job titles: {e}")
