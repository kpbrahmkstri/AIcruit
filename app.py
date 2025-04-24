import io
import os
import json
import streamlit as st
import pandas as pd
import pdfplumber
import docx
import openai
import base64
from streamlit.components.v1 import html
import altair as alt

from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

    job_titles = job_descriptions.distinct("job_title")
    if not job_titles:
        st.warning("No job descriptions found. Please upload some in Tab 2.")
    else:
        selected_title = st.selectbox("Select Job Title", sorted(job_titles), key="dashboard_title")

        # Fetch matched resumes for selected job title
        matches = resumes.find()
        matched_data = []

        for res in matches:
            parsed = res.get("parsed", {})
            if not parsed:
                continue

            matched_data.append({
                "Filename": res.get("filename", "N/A"),
                "Upload Time": res.get("upload_time"),
                "Soft Skills Score": res.get("soft_skills_score", 0),
                "Technical Skills Score": res.get("technical_skills_score", 0),
                "Presentation Score": res.get("presentation_score", 0),
                "Keyword Match Score": res.get("keyword_score", 0),
                "Overall Score": res.get("overall_score", 0),
                "Download Link": f"data:application/octet-stream;base64,{res.get('file_data').hex()}"
            })

        if not matched_data:
            st.info("No resumes matched or scored yet for this title.")
        else:
            df = pd.DataFrame(matched_data)
            df_sorted = df.sort_values(by="Overall Score", ascending=False).reset_index(drop=True)
            df_sorted.index += 1
            df_sorted.insert(0, "Rank", df_sorted.index)

            st.subheader("🏆 Leaderboard")
            st.dataframe(df_sorted[["Rank", "Filename", "Overall Score", "Soft Skills Score", "Technical Skills Score", "Presentation Score", "Keyword Match Score"]])

            # Bar Chart of Overall Scores
            st.subheader("📊 Score Distribution")
            bar_chart = alt.Chart(df_sorted).mark_bar().encode(
                x=alt.X("Filename", sort='-y'),
                y="Overall Score",
                color=alt.Color("Overall Score", scale=alt.Scale(scheme='blues')),
                tooltip=["Filename", "Overall Score"]
            ).properties(height=400)
            st.altair_chart(bar_chart, use_container_width=True)

            # Optional: Radar chart or radar-style scores for selected resume
            selected_candidate = st.selectbox("Analyze Candidate", df_sorted["Filename"].tolist(), key="candidate_analysis")
            selected_row = df_sorted[df_sorted["Filename"] == selected_candidate].iloc[0]

            st.markdown("### Resume Strength Radar")
            st.write({
                "Soft Skills": selected_row["Soft Skills Score"],
                "Technical Skills": selected_row["Technical Skills Score"],
                "Presentation": selected_row["Presentation Score"],
                "Keyword Match": selected_row["Keyword Match Score"]
            })

            # Download as CSV
            csv = df_sorted.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Leaderboard CSV", csv, "leaderboard.csv", "text/csv")
