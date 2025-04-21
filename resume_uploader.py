# resume_uploader.py

import io
import os
import json
import pdfplumber
import docx
import openai
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv
import streamlit as st

#─── ENV & CLIENT SETUP ─────────────────────────────────────────────────────────
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

mongo_uri = os.getenv("MONGO_URI")
client    = MongoClient(mongo_uri)
db        = client["aicruit"]
resumes   = db["resumes"]

#─── RAW TEXT EXTRACTION ────────────────────────────────────────────────────────
def extract_text(uploaded_file) -> str:
    """Detects file type and returns full resume text."""
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
        raise ValueError("Unsupported file type: " + uploaded_file.type)
    

def normalize_parsed(parsed: dict) -> dict:
    import json
    normalized = {}

    for key, value in parsed.items():
        if isinstance(value, str):
            normalized[key] = value.strip()
        elif isinstance(value, list):
            if not value:
                normalized[key] = ""
            elif all(isinstance(v, str) for v in value):
                normalized[key] = "\n".join(v.strip() for v in value)
            elif all(isinstance(v, dict) for v in value):
                normalized[key] = "\n\n".join(json.dumps(v, indent=2) for v in value)
            else:
                normalized[key] = json.dumps(value, indent=2)
        elif isinstance(value, dict):
            normalized[key] = json.dumps(value, indent=2)
        else:
            normalized[key] = str(value)
    return normalized


#─── STRUCTURED PARSING VIA GPT ─────────────────────────────────────────────────
def parse_with_gpt(resume_text: str) -> dict:
    """Ask GPT-4 to return JSON with the four target sections."""
    system = {
        "role": "system",
        "content": (
            "You are a JSON-output resume parser. "
            "Given a candidate's resume as plain text, "
            "extract and return a JSON object with exactly these keys: "
            "work_experience, education, skills, achievements. "
            "Each value should be a newline‑delimited string."
        )
    }
    user = {"role": "user", "content": resume_text}

    resp = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[system, user],
        temperature=0
    )
    payload = resp.choices[0].message.content.strip()
    # Ensure valid JSON
    return json.loads(payload)

#─── STREAMLIT UPLOAD HANDLER ──────────────────────────────────────────────────
def handle_upload():
    st.header("Upload & Parse Resume")
    uploaded_file = st.file_uploader("PDF or DOCX only", type=["pdf", "docx"])
    if not uploaded_file:
        return

    # 1) Extract text
    with st.spinner("Extracting text…"):
        raw_text = extract_text(uploaded_file)

    # 2) Parse with GPT
    with st.spinner("Parsing sections via GPT-4…"):
        try:
            parsed_raw = parse_with_gpt(raw_text)

            # ✅ Normalize to all strings before insert
            parsed = normalize_parsed(parsed_raw)

            # ✅ Confirm every field is now a string
            assert all(isinstance(v, str) for v in parsed.values()), "Parsed fields must be strings"

        except Exception as e:
            st.error(f"OpenAI error or normalization failed: {e}")
            return

    # ✅ Debug view
    st.subheader("💡 Normalized Output Preview:")
    st.json(parsed)

    # 3) Prepare MongoDB document
    doc = {
        "filename":    uploaded_file.name,
        "upload_time": datetime.utcnow(),
        "file_type":   uploaded_file.type,
        "file_data":   uploaded_file.getvalue(),
        "parsed":      parsed  # ← All values here are strings
    }

    # 4) Insert into MongoDB
    try:
        resumes.insert_one(doc)
        st.success("✅ Resume parsed and saved to MongoDB!")
    except Exception as e:
        st.error(f"❌ Mongo insert failed: {e}")


    # 4) Show result
    st.subheader("Parsed Output:")
    st.json(parsed)


#─── IF THIS FILE IS RUN DIRECTLY ───────────────────────────────────────────────
if __name__ == "__main__":
    st.set_page_config(page_title="AIcruit Resume Uploader", layout="wide")
    handle_upload()
