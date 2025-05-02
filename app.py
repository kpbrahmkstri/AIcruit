import io
import os
import json
import re
import streamlit as st
import pandas as pd
import pdfplumber
import docx
import openai
import base64
from dateutil import parser as date_parser
from streamlit.components.v1 import html
import altair as alt

from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

#─── ENV & CLIENT SETUP ─────────────────────────────────────────────────────────
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

mongo_uri = os.getenv("MONGO_URI")
client    = MongoClient(mongo_uri)
db        = client["aicruit"]
resumes   = db["resumes"]
job_descriptions = db["job_descriptions"]
evaluations = db["evaluations"]

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

def extract_json_from_response(text):
    try:
        # Find JSON content between possible markdown backticks
        json_match = re.search(r'```(?:json)?(.*?)```', text, re.DOTALL)
        if json_match:
            # Extract content from between backticks
            json_content = json_match.group(1).strip()
        else:
            # If no backticks, use the entire text
            json_content = text.strip()
            
        # Remove any non-JSON characters at beginning and end
        # Find first '{' and last '}'
        start_idx = json_content.find('{')
        end_idx = json_content.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            json_content = json_content[start_idx:end_idx+1]
            
        # Parse the JSON
        return json.loads(json_content)
    except Exception as e:
        print(f"❌ Failed to parse JSON: {e}")
        print(f"Problematic text: {text[:200]}...")  # Print first 200 chars for debugging
        return None

#----Matching function ---------------------------------------------------------

def evaluate_resume_with_gpt(jd_text, resume_text):
    prompt = f"""
Given the following job description:
{jd_text}

And this resume:
{resume_text}

Evaluate the resume for:
1. Keyword matches
2. Relevant experience
3. Skills alignment (technical & soft)
4. Overall presentation

Return JSON:
{{
  "keyword_score": int (0-10),
  "experience_score": int (0-10),
  "technical_skills_score": int (0-10),
  "soft_skills_score": int (0-10),
  "presentation_score": int (0-10),
  "overall_score": float (0-10),
  "match_reason": string,
  "missing_elements": string
}}
"""
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a resume reviewer."}, {"role": "user", "content": prompt}],
        temperature=0.7
    )
    return json.loads(response.choices[0].message.content.strip())


def download_link(data: bytes, filename: str, label: str) -> str:
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{label}</a>'
    return href


# Helper function for parsing date
def parse_date_or_none(dt_str):
    try:
        return date_parser.parse(dt_str, default=datetime(1900, 1, 1))
    except Exception:
        return None

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
tab1, tab2, tab3 = st.tabs([
    "📄 Applicant Portal", 
    "📄 Recruiter Portal", 
    "📊 Recruiter Dashboard"
])


# ─── LangChain Setup ───────────────────────────────────────────────────────────


llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

# 1) Resume Parser Chain
resume_parser_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a resume parser. Return valid JSON with these keys:\n"
     "1. work_experience: list of objects with keys job_title, company, start_date, end_date, description\n"
     "2. education: string\n"
     "3. skills: list of strings\n"
     "4. achievements: list of strings\n"
     "5. certifications: list of strings"
    ),
    ("user", "{resume_text}")
])

resume_parser_chain = LLMChain(llm=llm, prompt=resume_parser_prompt)

# Resume vs JD Evaluation Chain
evaluation_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an AI resume evaluator.  \n"
     "Given a resume and a job description, return valid JSON with these keys and types:  \n"
     "- matched_skills (array of strings)  \n"
     "- missing_skills (array of strings)  \n"
     "- matched_experience (array of strings)  \n"
     "- missing_experience (array of strings)  \n"
     "- matched_certifications (array of strings)  \n"
     "- missing_certifications (array of strings)  \n"
     "- skills_match_score (number between 0 and 100)  \n"
     "- experience_match_score (number between 0 and 100)  \n"
     "- education_match_score (number between 0 and 100)  \n"
     "- certification_match_score (number between 0 and 100)  \n"
     "- overall_score (number between 0 and 100)  \n"
     "- strengths (array of strings)  \n"
     "- weaknesses (array of strings)  \n"
     "Only output the JSON."
    ),
    ("user",
     "Resume Text:\n{resume_text}\n\n"
     "Job Description Text:\n{jd_text}\n\n"
     "Return the JSON as specified."
    )
])
evaluation_chain = LLMChain(llm=llm, prompt=evaluation_prompt)


def evaluate_resume_against_jd(resume_text, jd_text):
    try:
        response = evaluation_chain.run({
            "resume_text": resume_text,
            "jd_text": jd_text
        })
        parsed = json.loads(response)
        return parsed
    except Exception as e:
        print(f"Evaluation Error: {e}")
        return None


# === Tab 1: Upload, Parse, Save & Evaluate ===
with tab1:
    st.header("📄 Upload Resume")

    all_jds = list(job_descriptions.find())
    titles = [jd["job_title"] for jd in all_jds]
    selected_title = st.selectbox("Select Job Title", ["Select Job Title"] + titles)

    uploaded_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])

    if "last_uploaded_filename" not in st.session_state:
        st.session_state["last_uploaded_filename"] = None
        st.session_state["last_resume_id"] = None

    if uploaded_file and selected_title != "Select Job Title":
        selected_jd = next((jd for jd in all_jds if jd["job_title"] == selected_title), None)

        with st.spinner("Extracting resume text..."):
            resume_text = extract_text(uploaded_file)

        jd_parsed = selected_jd.get("parsed", {})
        jd_text = "\n".join([
            "\n".join(jd_parsed.get("responsibilities", [])),
            "\n".join(jd_parsed.get("required_skills", [])),
            "\n".join(jd_parsed.get("qualifications", [])),
            "\n".join(jd_parsed.get("certifications", []))
        ])

        evaluation_template = ChatPromptTemplate.from_template(
            """You are an AI resume evaluator.

            Given the following job description:
            {jd_text}

            And this resume:
            {resume_text}

            Return valid JSON with these keys and types:
            - first_name (string)
            - last_name (string)
            - email (string)
            - matched_skills (array of strings)
            - missing_skills (array of strings)
            - matched_experience (array of objects with job_title, company, start_date, end_date, description)
            - missing_experience (array of strings)
            - matched_certification (array of strings)
            - missing_certification (array of strings)
            - skills_match_score (number between 0 to 100)
            - experience_match_score (number between 0 to 100)
            - education_match_score (number between 0 to 100)
            - certification_match_score (number between 0 to 100)
            - overall_score (number between 0 to 100)
            - strengths (array of strings)
            - weaknesses (array of strings)

            IMPORTANT: Return ONLY valid JSON with no additional text or explanations.
            """
        )
        chain = LLMChain(llm=llm, prompt=evaluation_template)

        with st.spinner("Uploading..."):
            try:
                llm_response = chain.run({"resume_text": resume_text, "jd_text": jd_text})
                #st.expander("Debug: Raw LLM Response").text(llm_response)
                eval_output = extract_json_from_response(llm_response)

                if not eval_output:
                    st.error("❌ GPT-4 did not return valid JSON. Please check the raw response above.")
                    st.stop()
            except Exception as e:
                st.error(f"❌ Error calling LLM: {e}")
                st.stop()

        #st.subheader("AI Evaluation Output")
        #st.json(eval_output)

        # Construct parsed_resume object from evaluation
        parsed_resume = {
            "work_experience": eval_output.get("matched_experience", []),
            "education": eval_output.get("education", jd_parsed.get("qualifications", "")),
            "skills": eval_output.get("matched_skills", []),
            "achievements": [],
            "certifications": eval_output.get("matched_certification", [])
        }

        # Sanitize fields
        for field in ["work_experience", "skills", "achievements", "certifications"]:
            if not isinstance(parsed_resume.get(field), list):
                parsed_resume[field] = []

        education_raw = parsed_resume.get("education", "")
        if isinstance(education_raw, list):
            education_raw = ", ".join([str(e) for e in education_raw])
        elif not isinstance(education_raw, str):
            education_raw = str(education_raw)
        parsed_resume["education"] = education_raw

        cleaned_experience = []
        for job in parsed_resume["work_experience"]:
            if isinstance(job, dict):
                job["start_date"] = parse_date_or_none(job.get("start_date", ""))
                job["end_date"] = parse_date_or_none(job.get("end_date", ""))
                cleaned_experience.append(job)
        parsed_resume["work_experience"] = cleaned_experience

        # Prepare highlights as strings for evaluation schema
        highlights_str = []
        for item in parsed_resume["work_experience"]:
            parts = [
                f"{item.get('job_title', '')} at {item.get('company', '')}",
                f"{item.get('start_date', '')} to {item.get('end_date', '')}",
                item.get("description", "")
            ]
            highlights_str.append(" | ".join([p for p in parts if p.strip()]))

        if uploaded_file.name != st.session_state["last_uploaded_filename"]:
            resume_doc = {
                "filename": uploaded_file.name,
                "upload_time": datetime.utcnow(),
                "file_type": uploaded_file.type,
                "file_data": uploaded_file.getvalue(),
                "job_id": selected_jd["_id"],
                "first_name": eval_output.get("first_name", ""),
                "last_name": eval_output.get("last_name", ""),
                "email": eval_output.get("email", ""),
                "parsed": parsed_resume
            }
            res_insert = resumes.insert_one(resume_doc)
            resume_id = res_insert.inserted_id
            st.session_state["last_uploaded_filename"] = uploaded_file.name
            st.session_state["last_resume_id"] = resume_id
            st.success(f"✅ Resume saved successfully (ID: {resume_id})")
        else:
            resume_id = st.session_state["last_resume_id"]
            st.info(f"ℹ️ Resume already uploaded this session. Using ID: {resume_id}")

        existing_eval = evaluations.find_one({
            "resumeId": resume_id,
            "jobId": selected_jd["_id"]
        })

        if existing_eval:
            st.info("ℹ️ Evaluation for this resume and JD already exists.")
        else:
            eval_doc = {
                "resumeId": resume_id,
                "jobId": selected_jd["_id"],
                "first_name": eval_output.get("first_name", ""),
                "last_name": eval_output.get("last_name", ""),
                "email": eval_output.get("email", ""),
                "overallScore": float(eval_output.get("overall_score", 0)),
                "categoryScores": {
                    "skillsMatch": float(eval_output.get("skills_match_score", 0)),
                    "experienceMatch": float(eval_output.get("experience_match_score", 0)),
                    "educationMatch": float(eval_output.get("education_match_score", 0)),
                    "certificationMatch": float(eval_output.get("certification_match_score", 0)),
                },
                "matchDetails": {
                    "matchedSkills": eval_output.get("matched_skills", []),
                    "missingSkills": eval_output.get("missing_skills", []),
                    "relevantExperience": {
                        "score": float(eval_output.get("experience_match_score", 0)),
                        "highlights": highlights_str
                    },
                    "educationAlignment": {
                        "score": float(eval_output.get("education_match_score", 0)),
                        "comments": ""
                    }
                },
                "feedback": {
                    "strengths": eval_output.get("strengths", []),
                    "weaknesses": eval_output.get("weaknesses", []),
                    "improvementSuggestions": eval_output.get("missing_skills", [])
                },
                "llmAnalysis": "Evaluation generated using GPT-4 (LangChain).",
                "llmModel": "GPT-4",
                "evaluationDate": datetime.utcnow()
            }
            ev_insert = evaluations.insert_one(eval_doc)
            st.success(f"✅ Evaluation saved successfully (ID: {ev_insert.inserted_id})")

    elif uploaded_file:
        st.warning("⚠️ Please select a valid Job Title before uploading a resume.")
    else:
        st.info("📥 Upload a resume and select a job title to begin.")

# 1) LLM client
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

# 2) Prompt template for JD parsing (responsibilities, required_skills, qualifications, certifications)
jd_parser_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a job description parser. Extract and return valid JSON with keys: responsibilities, required_skills, qualifications, certifications."),
    ("user", "{jd_text}")
])

# 3) Build the chain
jd_parser_chain = LLMChain(llm=llm, prompt=jd_parser_prompt)


# === TAB 2: Job Description Upload & Parsing (LangChain) ===
with tab2:
    st.header("📄 Job Description Upload")
    st.write("Enter job title + description or upload a JD file")

    # --- Session State Init (before widgets) ---
    if "job_title" not in st.session_state:
        st.session_state["job_title"] = ""
    if "job_desc_input" not in st.session_state:
        st.session_state["job_desc_input"] = ""

    # --- Widgets ---
    job_title = st.text_input(
        "Job Title",
        value=st.session_state["job_title"],
        key="job_title",
        placeholder="e.g., Data Scientist"
    )

    jd_file = st.file_uploader(
        "Or Upload JD in PDF/DOCX",
        type=["pdf", "docx"],
        key="jd_file"
    )

    # --- Extract text if file uploaded ---
    jd_text = ""
    if jd_file:
        with st.spinner("Extracting text…"):
            jd_text = extract_text(jd_file)

    job_desc_input = st.text_area(
        "Job Description",
        value=jd_text if jd_text else st.session_state["job_desc_input"],
        key="job_desc_input",
        height=250
    )

    # --- Save Logic ---
    if st.button("Save Job Description"):
        if not job_title.strip():
            st.warning("Please enter a job title.")
        elif not job_desc_input.strip():
            st.warning("Please enter or upload a job description.")
        else:
            try:
                with st.spinner("Uploading…"):
                    chain_output = jd_parser_chain.run({"jd_text": job_desc_input.strip()})
                    parsed_jd = extract_json_from_response(chain_output)

                if not parsed_jd:
                    st.error("❌ Parsing failed: No valid JSON from LLM.")
                    st.stop()

                parsed_jd.setdefault("certifications", [])

                jd_doc = {
                    "job_title":   job_title.strip(),
                    "upload_time": datetime.utcnow(),
                    "file_type":   jd_file.type if jd_file else "text",
                    "filename":    jd_file.name if jd_file else "manual_entry",
                    "file_data":   jd_file.getvalue() if jd_file else None,
                    "description": job_desc_input.strip(),
                    "parsed": {
                        "responsibilities": parsed_jd.get("responsibilities", []),
                        "required_skills":  parsed_jd.get("required_skills", []),
                        "qualifications":   parsed_jd.get("qualifications", []),
                        "certifications":   parsed_jd.get("certifications", [])
                    }
                }

                result = job_descriptions.insert_one(jd_doc)
                st.success(f"✅ Saved JD with ID: {result.inserted_id}")

                # --- Clean up and rerun ---
                st.session_state.pop("job_title", None)
                st.session_state.pop("job_desc_input", None)
                st.experimental_rerun()

            except Exception as e:
                st.error(f"❌ Error: {e}")


# Tab 3: Matching 
import base64

# Helper function to create a download link for each resume
def generate_download_link(file_data, filename, label="Download Resume"):
    b64 = base64.b64encode(file_data).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{label}</a>'


# ─── Tab 3: Top Resume Matching + Scatter Plot ───────────────────────────────
with tab3:
    st.header("📊 Top Resume Matching Dashboard")

    # 1) Fetch all Job Titles
    jd_list = list(job_descriptions.find())
    jd_title_list = ["Select Job Title"] + [jd["job_title"] for jd in jd_list]

    selected_title = st.selectbox("Select Job Title to View Resumes", jd_title_list)

    if selected_title != "Select Job Title":
        selected_jd = next((jd for jd in jd_list if jd["job_title"] == selected_title), None)

        if selected_jd:
            # 2) Fetch all resumes and evaluations for this JD
            matching_resumes = list(resumes.find({"job_id": selected_jd["_id"]}))

            if matching_resumes:
                resume_ids = [res["_id"] for res in matching_resumes]
                evals = list(evaluations.find({"resumeId": {"$in": resume_ids}}))

                rows = []
                scatter_data = []

                for res in matching_resumes:
                    eval_entry = next((ev for ev in evals if ev["resumeId"] == res["_id"]), None)

                    if not eval_entry:
                        continue

                    rows.append({
                        "First Name": res.get("first_name", ""),
                        "Last Name": res.get("last_name", ""),
                        "Email": res.get("email", ""),
                        "Filename": res["filename"],
                        "Download": generate_download_link(res["file_data"], res["filename"]),
                        "% Match (Overall)": round(eval_entry.get("overallScore", 0), 2),
                        "Skills Match": round(eval_entry.get("categoryScores", {}).get("skillsMatch", 0), 2),
                        "Experience Match": round(eval_entry.get("categoryScores", {}).get("experienceMatch", 0), 2),
                        "Education Match": round(eval_entry.get("categoryScores", {}).get("educationMatch", 0), 2),
                        "Certification Match": round(eval_entry.get("categoryScores", {}).get("certificationMatch", 0), 2),
                        "Missing Skills": ", ".join(eval_entry.get("matchDetails", {}).get("missingSkills", [])),
                        "Strengths": ", ".join(eval_entry.get("feedback", {}).get("strengths", [])),
                        "Weaknesses": ", ".join(eval_entry.get("feedback", {}).get("weaknesses", [])),
                        "Upload Time": res["upload_time"].strftime("%Y-%m-%d %H:%M:%S")
                    })

                    scatter_data.append({
                        "Name": f"{res.get('first_name', '')} {res.get('last_name', '')}".strip() or res["filename"],
                        "Skills Match": eval_entry.get("categoryScores", {}).get("skillsMatch", 0),
                        "Experience Match": eval_entry.get("categoryScores", {}).get("experienceMatch", 0),
                        "Type": "Resume"
                    })

                rows = sorted(rows, key=lambda x: x["% Match (Overall)"], reverse=True)

                for idx, row in enumerate(rows, 1):
                    row["Rank"] = idx

                columns_order = [
                    "Rank", "First Name", "Last Name", "Email",
                    "Filename", "Download",
                    "% Match (Overall)", "Skills Match", "Experience Match",
                    "Education Match", "Certification Match",
                    "Missing Skills", "Strengths", "Weaknesses", "Upload Time"
                ]
                rows_display = [{col: r.get(col, "") for col in columns_order} for r in rows]

                st.subheader(f"🏆 Top Resumes for: {selected_title}")
                st.write(
                    pd.DataFrame(rows_display).to_html(escape=False, index=False),
                    unsafe_allow_html=True
                )

                if scatter_data:
                    scatter_data.append({
                        "Name": f"⭐ JD - {selected_title}",
                        "Skills Match": 100,
                        "Experience Match": 100,
                        "Type": "Job Description"
                    })

                    scatter_df = pd.DataFrame(scatter_data)

                    scatter_chart = alt.Chart(scatter_df).mark_circle(size=130).encode(
                        x=alt.X('Skills Match:Q', scale=alt.Scale(domain=[0, 110])),
                        y=alt.Y('Experience Match:Q', scale=alt.Scale(domain=[0, 110])),
                        color='Type:N',
                        tooltip=['Name', 'Skills Match', 'Experience Match']
                    ).interactive()

                    st.subheader("📈 Resume vs JD: Skills vs Experience")
                    st.altair_chart(scatter_chart, use_container_width=True)
            else:
                st.info("No resumes uploaded for this job description yet.")
        else:
            st.error("Selected Job Description not found!")
    else:
        st.info("Please select a Job Title first to view matching resumes.")
