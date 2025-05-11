import io
import os
import json
import base64
import pdfplumber
import docx
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from datetime import datetime
from dateutil import parser as date_parser
from dotenv import load_dotenv
import openai

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# ─── ENV & CLIENT SETUP ─────────────────────────────────────────────────────────
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

client = MongoClient(os.getenv("MONGO_URI"))
db = client["aicruit"]
resumes = db["resumes"]
job_descriptions = db["job_descriptions"]
evaluations = db["evaluations"]

# ─── FastAPI Setup ──────────────────────────────────────────────────────────────
app = FastAPI(title="AIcruit Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Helpers ─────────────────────────────────────────────────────────────────────
def extract_text_from_bytes(data: bytes, filename: str) -> str:
    if filename.endswith(".pdf"):
        text = ""
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        if not text.strip():
            raise HTTPException(status_code=400, detail="Failed to extract text from PDF.")
        return text

    elif filename.endswith(".docx"):
        doc = docx.Document(io.BytesIO(data))
        text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        if not text.strip():
            raise HTTPException(status_code=400, detail="Failed to extract text from DOCX.")
        return text

    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

def parse_date_or_none(dt_str):
    try:
        return date_parser.parse(dt_str, default=datetime(1900, 1, 1))
    except Exception:
        return None

def extract_json_from_response(text: str):
    try:
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx == -1 or end_idx == -1:
            return None
        return json.loads(text[start_idx:end_idx+1])
    except Exception as e:
        print("❌ Failed to parse JSON:", e)
        return None

# ─── LangChain GPT Chains ─────────────────────────────────────────────────────────
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

evaluation_prompt = ChatPromptTemplate.from_template(
    """You are an AI resume evaluator. \n"
    Given the following job description:
    {jd_text}
    And this resume:
    {resume_text}
    Return valid JSON with these keys and types: \n "
    "- first_name (string)\n"
    "- last_name (string)\n"
    "- email (string)\n"
    "- matched_skills (array of strings) \n"
    "- missing_skills (array of strings) \n"
    "- matched_experience (list of objects: job_title, company, start_date, end_date, description) \n"
    "- missing_experience  (array of strings) \n"
    "- matched_certification  (array of strings) \n"
    "- missing_certification  (array of strings) \n"
    "- skills_match_score (number between 0 and 100) \n"
    "- keyword_score (number between 0 and 100) \n"
    "- experience_match_score (number between 0 and 100) \n"
    "- education_match_score (number between 0 and 100) \n"
    "- certification_match_score (number between 0 and 100) \n"
    "- overall_score (number between 0 and 100) \n"
    "- strengths  (array of strings) \n"
    "- weaknesses  (array of strings) \n"
    
    Return JSON:
    {{
    "keyword_score": int(0-10),
    "experience_score": int(0-10),
    "technical_skills_score": int(0-10),
    "soft_skills_score": int(0-10),
    "presentation_score":int(0-10),
    "overall_score": int(0-10),
    "match_reason" string,
    "missing_elements": string,
    "-matched_skills (array of strings) \n"
    "-missing_skills (array of strings) \n"
    "-matched_experience (array of strings) \n"
    "-missing_experience (array of strings) \n"
    "-matched_certifications (array of strings) \n"
    "-missing_certifications (array of strings) \n"
    "-skills_match_score (number between 0 and 100) \n"
    "-experience_match_score (number between 0 and 100) \n"
    "-education_match_score (number between 0 and 100) \n"
    "-certification_match_score (number between 0 and 100) \n"
    "-overall_score (number between 0 and 100) \n"
    "-strengths (array of strings) \n"
    "-weaknesses (array of strings) \n"
    }}
    """
)
evaluation_chain = LLMChain(llm=llm, prompt=evaluation_prompt)

jd_parser_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a job description parser. Extract and return valid JSON with keys: responsibilities, required_skills, qualifications, certifications."),
    ("user", "{jd_text}")
])
jd_parser_chain = LLMChain(llm=llm, prompt=jd_parser_prompt)

# ─── Health Check ────────────────────────────────────────────────────────────────
@app.get("/")
def read_root():
    return {"message": "AIcruit API is running."}

# ─── Endpoint: Upload Resume ─────────────────────────────────────────────────────
@app.post("/upload-resume/")
async def upload_resume(job_title: str = Form(...), file: UploadFile = File(...)):
    data = file.file.read()
    resume_text = extract_text_from_bytes(data, file.filename.lower())

    jd = job_descriptions.find_one({"job_title": job_title})
    if not jd:
        raise HTTPException(status_code=404, detail="Job title not found in database")

    jd_parsed = jd.get("parsed", {})
    jd_text = "\n".join([
        "\n".join(jd_parsed.get("responsibilities", [])),
        "\n".join(jd_parsed.get("required_skills", [])),
        "\n".join(jd_parsed.get("qualifications", [])),
        "\n".join(jd_parsed.get("certifications", []))
    ])

    try:
        llm_response = evaluation_chain.run({"resume_text": resume_text, "jd_text": jd_text})
        parsed = extract_json_from_response(llm_response)
        if not parsed:
            raise Exception("Invalid GPT response format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    parsed_resume = {
        "work_experience": parsed.get("matched_experience", []),
        "education": parsed.get("education", ""),
        "skills": parsed.get("matched_skills", []),
        "achievements": [],
        "certifications": parsed.get("matched_certification", [])
    }
    if isinstance(parsed_resume["education"], list):
        parsed_resume["education"] = ", ".join(parsed_resume["education"])

    cleaned_experience = []
    for job in parsed_resume["work_experience"]:
        if isinstance(job, dict):
            job["start_date"] = parse_date_or_none(job.get("start_date", ""))
            job["end_date"] = parse_date_or_none(job.get("end_date", ""))
            cleaned_experience.append(job)
    parsed_resume["work_experience"] = cleaned_experience

    highlights_str = []
    for item in cleaned_experience:
        parts = [
            f"{item.get('job_title', '')} at {item.get('company', '')}",
            f"{item.get('start_date', '')} to {item.get('end_date', '')}",
            item.get("description", "")
        ]
        highlights_str.append(" | ".join([p for p in parts if p.strip()]))

    resume_doc = {
        "filename": file.filename,
        "upload_time": datetime.utcnow(),
        "file_type": file.content_type,
        "file_data": data,
        "job_id": jd["_id"],
        "first_name": parsed.get("first_name", ""),
        "last_name": parsed.get("last_name", ""),
        "email": parsed.get("email", ""),
        "parsed": parsed_resume
    }
    res_id = resumes.insert_one(resume_doc).inserted_id

    eval_doc = {
        "resumeId": res_id,
        "jobId": jd["_id"],
        "first_name": parsed.get("first_name", ""),
        "last_name": parsed.get("last_name", ""),
        "email": parsed.get("email", ""),
        "overallScore": float(parsed.get("overall_score", 0)),
        "categoryScores": {
            "skillsMatch": float(parsed.get("skills_match_score", 0)),
            "experienceMatch": float(parsed.get("experience_match_score", 0)),
            "educationMatch": float(parsed.get("education_match_score", 0)),
            "certificationMatch": float(parsed.get("certification_match_score", 0)),
            "keywordMatch": float(parsed.get("keyword_score", 0) * 10)  # assuming keyword_score is 0-10
        },
        "matchDetails": {
            "matchedSkills": parsed.get("matched_skills", []),
            "missingSkills": parsed.get("missing_skills", []),
            "relevantExperience": {
                "score": float(parsed.get("experience_match_score", 0)),
                "highlights": highlights_str
            },
            "educationAlignment": {
                "score": float(parsed.get("education_match_score", 0)),
                "comments": ""
            }
        },
        "feedback": {
            "strengths": parsed.get("strengths", []),
            "weaknesses": parsed.get("weaknesses", []),
            "improvementSuggestions": parsed.get("missing_skills", [])
        },
        "llmAnalysis": "Evaluation via GPT-4 (LangChain)",
        "llmModel": "GPT-4",
        "evaluationDate": datetime.utcnow()
    }
    evaluations.insert_one(eval_doc)

    return {"message": "✅ Resume and evaluation saved", "resume_id": str(res_id)}

# ─── Endpoint: Upload JD ─────────────────────────────────────────────────────────
@app.post("/upload-jd/")
async def upload_jd(job_title: str = Form(...), job_desc_input: str = Form(""), file: UploadFile = File(None)):
    try:
        jd_text = job_desc_input.strip()
        file_data = None
        filename = "manual_entry"
        content_type = "text"

        if file:
            file_data = await file.read()
            filename = file.filename
            content_type = file.content_type
            jd_text = extract_text_from_bytes(file_data, filename.lower())

        if not jd_text.strip():
            raise HTTPException(status_code=400, detail="Job description is empty.")

        chain_output = jd_parser_chain.run({"jd_text": jd_text})
        parsed_jd = extract_json_from_response(chain_output)
        if not parsed_jd:
            raise HTTPException(status_code=500, detail="Failed to parse JD")

        parsed_jd.setdefault("certifications", [])
        jd_doc = {
            "job_title": job_title.strip(),
            "upload_time": datetime.utcnow(),
            "file_type": content_type,
            "filename": filename,
            "file_data": file_data,
            "description": jd_text,
            "parsed": {
                "responsibilities": parsed_jd.get("responsibilities", []),
                "required_skills": parsed_jd.get("required_skills", []),
                "qualifications": parsed_jd.get("qualifications", []),
                "certifications": parsed_jd.get("certifications", [])
            }
        }
        result = job_descriptions.insert_one(jd_doc)
        return {"message": "✅ JD saved", "jd_id": str(result.inserted_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload JD error: {e}")

# ─── Endpoint: Dashboard ─────────────────────────────────────────────────────────
@app.get("/dashboard/{job_title}")
def get_dashboard(job_title: str):
    jd = job_descriptions.find_one({"job_title": job_title})
    if not jd:
        return {"rows": [], "scatter": []}
    resumes_matched = list(resumes.find({"job_id": jd["_id"]}))
    resume_ids = [r["_id"] for r in resumes_matched]
    evals = list(evaluations.find({"resumeId": {"$in": resume_ids}}))

    rows, scatter = [], []
    for res in resumes_matched:
        ev = next((e for e in evals if e["resumeId"] == res["_id"]), None)
        if not ev:
            continue
        rows.append({
            "First Name": res.get("first_name", ""),
            "Last Name": res.get("last_name", ""),
            "Email": res.get("email", ""),
            "Resume": f'<a href="data:application/octet-stream;base64,{base64.b64encode(res["file_data"]).decode()}" download="{res["filename"]}">Download</a>',
            "Match Overall": round(ev.get("overallScore", 0), 2),
            "Skills Match": round(ev.get("categoryScores", {}).get("skillsMatch", 0), 2),
            "Experience Match": round(ev.get("categoryScores", {}).get("experienceMatch", 0), 2),
            "Education Match": round(ev.get("categoryScores", {}).get("educationMatch", 0), 2),
            "Certification Match": round(ev.get("categoryScores", {}).get("certificationMatch", 0), 2),
            "Keyword Match": round(ev.get("categoryScores", {}).get("keywordMatch", 0), 2),
            "Missing Skills": ", ".join(ev.get("matchDetails", {}).get("missingSkills", [])),
            "Strengths": ", ".join(ev.get("feedback", {}).get("strengths", [])),
            "Weaknesses": ", ".join(ev.get("feedback", {}).get("weaknesses", [])),
            "Upload Time": res["upload_time"].strftime("%Y-%m-%d %H:%M:%S")
        })
        scatter.append({
            "Name": f"{res.get('first_name', '')} {res.get('last_name', '')}".strip() or res["filename"],
            "Skills Match": ev.get("categoryScores", {}).get("skillsMatch", 0),
            "Experience Match": ev.get("categoryScores", {}).get("experienceMatch", 0),
            "Type": "Resume"
        })

    scatter.append({
        "Name": f"⭐ JD - {job_title}",
        "Skills Match": 100,
        "Experience Match": 100,
        "Type": "Job Description"
    })

    return {"rows": rows, "scatter": scatter}

# ─── Endpoint: Get Job Titles ───────────────────────────────────────────────────
@app.get("/job-titles/")
def get_job_titles():
    titles = job_descriptions.distinct("job_title")
    return {"titles": sorted(titles)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)