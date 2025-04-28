import io
import os
import json
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

#─── STRUCTURED PARSING VIA GPT ─────────────────────────────────────────────────
def parse_with_gpt_old(resume_text: str) -> dict:
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

def parse_with_gpt(resume_text: str) -> dict:
    system = {"role": "system", "content": (
        "You are a JSON-output resume parser. Given a resume text, extract and return JSON with keys: work_experience, education, skills, achievements."
    )}
    user = {"role": "user", "content": resume_text}
    resp = openai.ChatCompletion.create(model="gpt-4", messages=[system, user], temperature=0)
    return json.loads(resp.choices[0].message.content.strip())


# ─── STRUCTURED JOB DESCRIPTION PARSING VIA GPT ────────────────────────────────
def parse_jd_with_gpt_old(jd_text: str) -> dict:
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

# Job Description Parser
def parse_jd_with_gpt(jd_text: str) -> dict:
    system = {"role": "system", "content": (
        "You are a job description parser. Extract and return a JSON with fields: responsibilities, required_skills, qualifications."
    )}
    user = {"role": "user", "content": jd_text}
    response = openai.ChatCompletion.create(model="gpt-4", messages=[system, user], temperature=0)
    return json.loads(response.choices[0].message.content.strip())

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
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a resume reviewer."}, {"role": "user", "content": prompt}],
        temperature=0
    )
    return json.loads(response.choices[0].message.content.strip())

def find_top_resumes_for_job_old(job_title: str, top_n: int = 5):

    # Get the selected JD document
    jd_doc = job_descriptions.find_one({"job_title": job_title})
    if not jd_doc:
        return [], "Job description not found."

    jd_parsed = jd_doc.get("parsed", {})
    jd_text = " ".join([
        "\n".join(jd_parsed.get("responsibilities", [])),
        "\n".join(jd_parsed.get("required_skills", [])),
        "\n".join(jd_parsed.get("qualifications", []))
    ])

    # Get all resumes
    all_resumes = list(resumes.find())

    # Prepare data for comparison
    scored_resumes = []
    for resume in all_resumes:
        parsed = resume.get("parsed", {})
        resume_text = " ".join([
            parsed.get("work_experience", ""),
            parsed.get("education", ""),
            parsed.get("skills", ""),
            parsed.get("achievements", "")
        ])

        # Vectorize both
        vectorizer = TfidfVectorizer().fit([jd_text, resume_text])
        vectors = vectorizer.transform([jd_text, resume_text])
        score = cosine_similarity(vectors[0], vectors[1])[0][0]

        # Get explanation via GPT
        explanation_prompt = f"""
        JD:
        {jd_text}

        RESUME:
        {resume_text}

        Analyze how well this resume matches the job description.
        Return a JSON with:
        - match_reason: why it scored well
        - missing_elements: what important aspects are missing
        """
        try:
            gpt_resp = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a resume reviewer."},
                    {"role": "user", "content": explanation_prompt}
                ],
                temperature=0
            )
            explanation = json.loads(gpt_resp.choices[0].message.content)
        except:
            explanation = {
                "match_reason": "N/A (GPT error)",
                "missing_elements": "N/A (GPT error)"
            }

        scored_resumes.append({
            "Filename": resume.get("filename"),
            "Score": round(score * 100, 2),
            "Match Reason": explanation["match_reason"],
            "Missing Compared to JD": explanation["missing_elements"]
        })

    # Sort and rank
    sorted_resumes = sorted(scored_resumes, key=lambda r: r["Score"], reverse=True)
    for idx, r in enumerate(sorted_resumes, start=1):
        r["Rank"] = idx
        #r["Sr. No"] = idx

    return sorted_resumes[:top_n], None


def download_link(data: bytes, filename: str, label: str) -> str:
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{label}</a>'
    return href

# Find top resumes based on GPT scoring
def find_top_resumes_for_job(job_title: str, top_n: int = 5):
    jd_doc = job_descriptions.find_one({"job_title": job_title})
    if not jd_doc:
        return [], "Job description not found."

    jd_parsed = jd_doc.get("parsed", {})
    jd_text = " ".join(["\n".join(jd_parsed.get(k, [])) for k in ["responsibilities", "required_skills", "qualifications"]])
    all_resumes = list(resumes.find())

    results = []
    for resume in all_resumes:
        parsed = resume.get("parsed", {})
        resume_text = " ".join([parsed.get(k, "") for k in ["work_experience", "education", "skills", "achievements"]])
        try:
            eval = evaluate_resume_with_gpt(jd_text, resume_text)
            results.append({
                #"Sr. No": len(results) + 1,
                "Filename": resume.get("filename"),
                "Keyword Match": eval["keyword_score"],
                "Experience": eval["experience_score"],
                "Tech Skills": eval["technical_skills_score"],
                "Soft Skills": eval["soft_skills_score"],
                "Presentation": eval["presentation_score"],
                "Overall Score": round(eval["overall_score"], 2),
                "Why it Scored High": eval["match_reason"],
                "Missing Compared to JD": eval["missing_elements"],
                "Download": resume.get("filename"),
            })
        except Exception as e:
            continue

    sorted_results = sorted(results, key=lambda x: x["Overall Score"], reverse=True)
    for idx, r in enumerate(sorted_results, 1):
        r["Rank"] = idx
    return sorted_results[:top_n], None

def find_top_resumes_for_job2(job_title: str, top_n: int = 5):
    jd_doc = job_descriptions.find_one({"job_title": job_title})
    if not jd_doc:
        return [], "Job description not found."

    jd_parsed = jd_doc.get("parsed", {})
    jd_text = " ".join([
        "\n".join(jd_parsed.get("responsibilities", [])),
        "\n".join(jd_parsed.get("required_skills", [])),
        "\n".join(jd_parsed.get("qualifications", []))
    ])
    tech_skills_text = ", ".join(jd_parsed.get("required_skills", []))
    soft_skills_prompt = f"Extract soft skills from this job description:\n{jd_text}\nReturn a list of important soft skills only."

    try:
        gpt_soft_resp = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": soft_skills_prompt}],
            temperature=0
        )
        soft_skills = json.loads(gpt_soft_resp.choices[0].message.content)
    except:
        soft_skills = ["Communication", "Teamwork", "Problem-solving"]

    all_resumes = list(resumes.find())
    scored_resumes = []

    for idx, resume in enumerate(all_resumes, 1):
        parsed = resume.get("parsed", {})
        resume_text = " ".join([
            parsed.get("work_experience", ""),
            parsed.get("education", ""),
            parsed.get("skills", ""),
            parsed.get("achievements", "")
        ])

        # TF-IDF score
        vectorizer = TfidfVectorizer().fit([jd_text, resume_text])
        vectors = vectorizer.transform([jd_text, resume_text])
        tfidf_score = cosine_similarity(vectors[0], vectors[1])[0][0]

        # GPT Matching Reason
        explanation_prompt = f"""
        JD:
        {jd_text}

        RESUME:
        {resume_text}

        Analyze the match between JD and resume.
        Return JSON:
        - match_reason
        - missing_elements
        - soft_skills_scores: dict of soft skill -> score /10
        - technical_skills_scores: dict of tech skill -> score /10
        - overall_score: out of 10
        """
        try:
            gpt_resp = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "system", "content": "You are a resume evaluator."},
                          {"role": "user", "content": explanation_prompt}],
                temperature=0
            )
            analysis = json.loads(gpt_resp.choices[0].message.content)
        except Exception as e:
            analysis = {
                "match_reason": "N/A",
                "missing_elements": "N/A",
                "soft_skills_scores": {},
                "technical_skills_scores": {},
                "overall_score": round(tfidf_score * 10, 2)
            }

        scored_resumes.append({
            #"Sr. No": idx,
            "Filename": resume["filename"],
            "Download": download_link(resume["file_data"], resume["filename"], "⬇️ Download"),
            "Soft Skills (from JD)": ", ".join(soft_skills),
            "Soft Skills Score (/10)": sum(analysis.get("soft_skills_scores", {}).values()) / len(soft_skills) if soft_skills else 0,
            "Technical Skills (from JD)": tech_skills_text,
            "Technical Skills Score (/10)": sum(analysis.get("technical_skills_scores", {}).values()) / len(jd_parsed.get("required_skills", [])) if jd_parsed.get("required_skills") else 0,
            "Overall Score (/10)": round(analysis["overall_score"], 2),
            "Why it scored high": analysis.get("match_reason", "N/A"),
            "What is missing": analysis.get("missing_elements", "N/A"),
            "Rank": None  # to be assigned
        })

    sorted_resumes = sorted(scored_resumes, key=lambda r: r["Overall Score (/10)"], reverse=True)
    for rank, r in enumerate(sorted_resumes, 1):
        r["Rank"] = rank

    return sorted_resumes[:top_n], None

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
#tab1, tab2, tab3 = st.tabs(["📄 Resume Upload", "📄 Job Description Upload", "📊 Top Resume Matching"])
tab1, tab2, tab3, tab4 = st.tabs([
    "📄 Resume Upload", 
    "📄 Job Description Upload", 
    "📊 Top Resume Matching", 
    "📈 Dashboard & Analytics"
])


# ─── LangChain Setup ───────────────────────────────────────────────────────────
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

llm = ChatOpenAI(model_name="gpt-4", temperature=0)

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

# 2) Evaluation Chain
evaluation_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a resume evaluator. Given a parsed resume and a parsed job description, return a JSON matching this schema:\n"
               "{ resumeId: string, jobId: string, overallScore: number, categoryScores: {skillsMatch,experienceMatch,educationMatch,certificationMatch}, "
               "matchDetails: {matchedSkills,missingSkills,relevantExperience:{score,highlights},educationAlignment:{score,comments}}, "
               "feedback:{strengths,weaknesses,improvementSuggestions}, llmAnalysis: string, llmModel: string, evaluationDate: string }"),
    ("user", "Resume: {resume_parsed}\nJob: {jd_parsed}")
])
evaluation_chain = LLMChain(llm=llm, prompt=evaluation_prompt)

def parse_date_or_none(dt_str):
    """
    Try to parse a date string into a datetime.
    If it fails (e.g. “Current”), return None.
    """
    try:
        # supply default day for month/year inputs
        return date_parser.parse(dt_str, default=datetime(1900, 1, 1))
    except Exception:
        return None


# Tab 1: Upload & Parse
# ─── Tab 1: Upload, Parse, Save & Evaluate ────────────────────────────────────


# === Tab 1: Upload, Parse, Save & Evaluate ===
with tab1:
    st.header("Upload Resume & Evaluate")
    st.write("Select a job, upload a resume, then we parse, store, score, and evaluate.")

    # 1) Load job options
    all_jds = list(job_descriptions.find())
    titles = [jd["job_title"] for jd in all_jds]
    selected_title = st.selectbox("Select Job to Apply", titles)
    selected_jd = next(jd for jd in all_jds if jd["job_title"] == selected_title)

    # 2) File uploader
    uploaded_file = st.file_uploader("Upload PDF/DOCX Resume", type=["pdf", "docx"])
    if not uploaded_file:
        st.stop()

    # 3) Extract raw text
    with st.spinner("Extracting resume text…"):
        raw_text = extract_text(uploaded_file)

    # 4) Parse resume via LangChain
    with st.spinner("Parsing resume…"):
        parsed_json = resume_parser_chain.run({"resume_text": raw_text})
        resume_parsed = json.loads(parsed_json)
    st.subheader("Parsed Resume")
    st.json(resume_parsed)

    # 5) Normalize parsed data
    normalized = {
        "work_experience": resume_parsed.get("work_experience", []),
        "education":       resume_parsed.get("education", ""),
        "skills":          resume_parsed.get("skills", []),
        "achievements":    resume_parsed.get("achievements", []),
        "certifications":  resume_parsed.get("certifications", [])
    }
    st.subheader("Normalized for MongoDB")
    st.json(normalized)

    # 6) Convert date strings to datetime for each work_experience entry
    for entry in normalized["work_experience"]:
        if isinstance(entry, dict):
            entry["start_date"] = parse_date_or_none(entry.get("start_date", ""))
            entry["end_date"]   = parse_date_or_none(entry.get("end_date", ""))

    # 7) Save resume document to MongoDB
    resume_doc = {
        "filename":    uploaded_file.name,
        "upload_time": datetime.utcnow(),
        "file_type":   uploaded_file.type,
        "file_data":   uploaded_file.getvalue(),
        "job_id":      selected_jd["_id"],
        "parsed":      normalized
    }
    res_insert = resumes.insert_one(resume_doc)
    resume_id = res_insert.inserted_id
    st.success(f"✅ Resume saved with ID: {resume_id}")

    # 8) Compute overall match percentage via TF-IDF
    jd_parsed = selected_jd["parsed"]
    jd_text = " ".join([
        "\n".join(jd_parsed.get("responsibilities", [])),
        "\n".join(jd_parsed.get("required_skills", [])),
        "\n".join(jd_parsed.get("qualifications", [])),
        "\n".join(jd_parsed.get("certifications", []))
    ])
    resume_text = " ".join([
        # join normalized fields into one string
        "; ".join([str(work) for work in normalized["work_experience"]]),
        normalized["education"],
        ", ".join(normalized["skills"]),
        ", ".join(normalized["achievements"]),
        ", ".join(normalized["certifications"])
    ])
    vec = TfidfVectorizer().fit([jd_text, resume_text])
    v0, v1 = vec.transform([jd_text, resume_text])
    score = float(cosine_similarity(v0, v1)[0][0])
    percent_match = round(score * 100, 2)
    st.info(f"Overall Match: {percent_match}%")

    # 9) Build & save evaluation document
    eval_doc = {
        "resumeId":       resume_id,
        "jobId":          selected_jd["_id"],
        "overallScore":   percent_match,
        "categoryScores": {
            "skillsMatch":         percent_match,
            "experienceMatch":     percent_match,
            "educationMatch":      percent_match,
            "certificationMatch":  percent_match
        },
        "matchDetails": {
            "matchedSkills":       normalized["skills"],
            "missingSkills":       [],  # could diff vs jd_parsed["required_skills"]
            "relevantExperience": {
                "score":      percent_match,
                "highlights": []
            },
            "educationAlignment": {
                "score":    percent_match,
                "comments": ""
            }
        },
        "feedback": {
            "strengths":             [],
            "weaknesses":            [],
            "improvementSuggestions": []
        },
        "llmAnalysis":    f"Resume matches JD at {percent_match}%.",
        "llmModel":       "TF-IDF+Cosine",
        "evaluationDate": datetime.utcnow()
    }
    ev_insert = evaluations.insert_one(eval_doc)
    st.success(f"✅ Evaluation saved with ID: {ev_insert.inserted_id}")


# 1) LLM client
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

# 2) Prompt template for JD parsing (responsibilities, required_skills, qualifications, certifications)
jd_parser_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a job description parser. Extract and return valid JSON with keys: responsibilities, required_skills, qualifications, certifications."),
    ("user", "{jd_text}")
])

# 3) Build the chain
jd_parser_chain = LLMChain(llm=llm, prompt=jd_parser_prompt)


# === TAB 2: Job Description Upload (LangChain) ===

# === TAB 2: Job Description Upload & Parsing (LangChain) ===
# === TAB 2: Job Description Upload & Parsing (LangChain) ===
with tab2:
    st.header("📄 Job Description Upload & Parsing")
    st.write("Enter job title + description or upload a JD file; then LangChain will parse it.")

    # Initialize session state for text inputs
    if "job_title" not in st.session_state:
        st.session_state.job_title = ""
    if "job_desc_input" not in st.session_state:
        st.session_state.job_desc_input = ""

    # Widgets with explicit keys
    job_title = st.text_input(
        "Job Title",
        value=st.session_state.job_title,
        key="job_title",
        placeholder="e.g., Data Scientist"
    )
    jd_file = st.file_uploader(
        "Or Upload JD PDF/DOCX",
        type=["pdf", "docx"],
        key="jd_file"
    )

    # Extract text if file uploaded
    jd_text = ""
    if jd_file:
        with st.spinner("Extracting text…"):
            jd_text = extract_text(jd_file)

    job_desc_input = st.text_area(
        "Job Description",
        value=jd_text if jd_text else st.session_state.job_desc_input,
        key="job_desc_input",
        height=250
    )

    if st.button("Parse & Save Job Description"):
        if not job_title.strip():
            st.warning("Please enter a job title.")
        elif not job_desc_input.strip():
            st.warning("Please enter or upload a job description.")
        else:
            try:
                with st.spinner("LangChain parsing via GPT-4…"):
                    chain_output = jd_parser_chain.run({"jd_text": job_desc_input.strip()})
                    parsed_jd = json.loads(chain_output)

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

                # Reset only the text inputs
                st.session_state.job_title = ""
                st.session_state.job_desc_input = ""

                # Rerun to clear file_uploader visually
                st.experimental_rerun()

            except Exception as e:
                st.error(f"❌ Error: {e}")


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
                with st.spinner("Evaluating resumes..."):
                    results, error = find_top_resumes_for_job(selected_job_title)
                    if error:
                        st.error(error)
                    else:
                        df = pd.DataFrame(results)
                        st.dataframe(df)
    except Exception as e:
        st.error(f"Error loading job titles: {e}")

with tab4:
    st.header("📈 Resume Analytics Dashboard")

    # Fetch all job titles
    job_titles = job_descriptions.distinct("job_title")
    job_titles = ["Select Job Title..."] + sorted(job_titles)

    selected_title = st.selectbox("Select Job Title", job_titles, key="dashboard_title")

    if selected_title and selected_title != "Select Job Title...":
        with st.spinner("Loading resumes and evaluating..."):
            # Fetch and score resumes for the selected job title
            results, error = find_top_resumes_for_job(selected_title, top_n=50)

            if error:
                st.error(error)
            elif not results:
                st.info("No resumes found for this job title.")
            else:
                df = pd.DataFrame(results)

                st.subheader("🏆 Leaderboard")
                df_sorted = df.sort_values(by="Overall Score", ascending=False).reset_index(drop=True)
                df_sorted.index += 1
                if "Rank" in df_sorted.columns:
                    df_sorted = df_sorted.drop(columns=["Rank"])
                df_sorted.insert(0, "Rank", df_sorted.index)

                st.dataframe(df_sorted[[
                    "Rank", "Filename", "Overall Score",
                    "Soft Skills", 
                    "Tech Skills", 
                    "Presentation", 
                    "Keyword Match"
                ]])

                st.markdown("---")

                # Scatter plot
                st.subheader("📈 Resume Match Scatter Plot")

                # Create ideal JD reference point
                jd_reference = pd.DataFrame([{
                    "Filename": "⭐ Job Description (Ideal)",
                    "Tech Skills": 10,
                    "Soft Skills": 10,
                    "Overall Score": 10,
                    "Keyword Match": 10,
                    "Presentation": 10
                }])

                # Scatter for resumes
                scatter = alt.Chart(df_sorted).mark_circle(size=150).encode(
                    x=alt.X("Tech Skills:Q", title="Technical Skills Match (/10)"),
                    y=alt.Y("Soft Skills:Q", title="Soft Skills Match (/10)"),
                    color=alt.Color(
                        "Overall Score:Q",
                        scale=alt.Scale(domain=[5, 8, 10], range=["red", "yellow", "green"]),
                        legend=alt.Legend(title="Match Quality")
                    ),
                    size=alt.Size("Overall Score:Q", scale=alt.Scale(range=[100, 500])),
                    tooltip=["Filename", "Overall Score", "Keyword Match", "Tech Skills", "Soft Skills", "Presentation"]
                )

                # Star for JD
                star = alt.Chart(jd_reference).mark_point(
                    shape="star", 
                    size=500, 
                    color="gold"
                ).encode(
                    x=alt.X("Tech Skills:Q"),
                    y=alt.Y("Soft Skills:Q"),
                    tooltip=["Filename"]
                )

                # Labels for resumes
                text = alt.Chart(df_sorted).mark_text(
                    align='left',
                    baseline='middle',
                    dx=7
                ).encode(
                    x="Tech Skills:Q",
                    y="Soft Skills:Q",
                    text="Filename"
                )

                # Label for JD
                jd_text = alt.Chart(jd_reference).mark_text(
                    text="⭐ JD",
                    align="left",
                    baseline="middle",
                    dx=7
                ).encode(
                    x="Tech Skills:Q",
                    y="Soft Skills:Q"
                )

                # Final combined plot
                final_chart = alt.layer(
                    scatter,
                    star,
                    text,
                    jd_text
                ).resolve_scale(
                    x='shared',
                    y='shared'
                ).properties(
                    height=600,
                    title="Resume Match vs Job Description"
                )

                st.altair_chart(final_chart, use_container_width=True)

                # CSV download
                csv = df_sorted.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Leaderboard CSV", csv, "leaderboard.csv", "text/csv")

    else:
        st.info("Please select a job title from the dropdown to see the leaderboard and scatter plot.")
