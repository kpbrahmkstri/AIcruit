from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()
client = MongoClient(os.getenv("MONGO_URI"))
db = client["aicruit"]


resume_schema = {
    "validator": {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["filename", "upload_time", "file_type", "file_data", "parsed"],
            "properties": {
                "filename": {"bsonType": "string"},
                "upload_time": {"bsonType": "date"},
                "file_type": {"bsonType": "string"},
                "file_data": {"bsonType": "binData"},
                "first_name": {"bsonType": "string"},
                "last_name": {"bsonType": "string"},
                "email": {"bsonType": "string"},
                "parsed": {
                    "bsonType": "object",
                    "required": ["work_experience", "education", "skills", "achievements", "certifications"],
                    "properties": {
                        "work_experience": {
                            "bsonType": "array",
                            "items": {
                                "bsonType": "object",
                                "required": ["job_title", "company", "start_date", "end_date", "description"],
                                "properties": {
                                    "job_title": {"bsonType": "string"},
                                    "company": {"bsonType": "string"},
                                    "start_date": {"bsonType": "date"},
                                    "end_date": {"bsonType": ["date", "null"]},
                                    "description": {"bsonType": "string"}
                                }
                            }
                        },
                        "education": {"bsonType": "string"},
                        "skills": {"bsonType": "array", "items": {"bsonType": "string"}},
                        "achievements": {"bsonType": "array", "items": {"bsonType": "string"}},
                        "certifications": {"bsonType": "array", "items": {"bsonType": "string"}}
                    }
                }
            }
        }
    }
}



job_description_schema = {
    "validator": {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["job_title", "upload_time", "file_type", "description"],
            "properties": {
                "job_title": {"bsonType": "string"},
                "upload_time": {"bsonType": "date"},
                "file_type": {"bsonType": "string"},
                "filename": {"bsonType": "string"},
                "file_data": {"bsonType": ["binData", "null"]},
                "description": {"bsonType": "string"},
                "parsed": {
                    "bsonType": "object",
                    "required": ["responsibilities", "required_skills", "qualifications", "certifications"],
                    "properties": {
                        "responsibilities": {"bsonType": "array", "items": {"bsonType": "string"}},
                        "required_skills": {"bsonType": "array", "items": {"bsonType": "string"}},
                        "qualifications": {"bsonType": "array", "items": {"bsonType": "string"}},
                        "certifications": {"bsonType": "array", "items": {"bsonType": "string"}}
                    }
                }
            }
        }
    }
}


evaluations_schema = {
    "validator": {
        "$jsonSchema": {
            "bsonType": "object",
            "required": [
                "resumeId", "jobId", "overallScore", "categoryScores", 
                "matchDetails", "feedback", "llmAnalysis", "llmModel", "evaluationDate"
            ],
            "properties": {
                "resumeId": {"bsonType": "objectId"},
                "jobId": {"bsonType": "objectId"},
                "overallScore": {"bsonType": "double"},
                "first_name": {"bsonType": "string"},
                "last_name": {"bsonType": "string"},
                "email": {"bsonType": "string"},
                "categoryScores": {
                    "bsonType": "object",
                    "required": ["skillsMatch", "experienceMatch", "educationMatch", "certificationMatch"],
                    "properties": {
                        "skillsMatch": {"bsonType": "double"},
                        "experienceMatch": {"bsonType": "double"},
                        "educationMatch": {"bsonType": "double"},
                        "certificationMatch": {"bsonType": "double"}
                    }
                },
                "matchDetails": {
                    "bsonType": "object",
                    "properties": {
                        "matchedSkills": {"bsonType": "array", "items": {"bsonType": "string"}},
                        "missingSkills": {"bsonType": "array", "items": {"bsonType": "string"}},
                        "relevantExperience": {
                            "bsonType": "object",
                            "properties": {
                                "score": {"bsonType": "double"},
                                "highlights": {"bsonType": "array", "items": {"bsonType": "string"}}
                            }
                        },
                        "educationAlignment": {
                            "bsonType": "object",
                            "properties": {
                                "score": {"bsonType": "double"},
                                "comments": {"bsonType": "string"}
                            }
                        }
                    }
                },
                "feedback": {
                    "bsonType": "object",
                    "properties": {
                        "strengths": {"bsonType": "array", "items": {"bsonType": "string"}},
                        "weaknesses": {"bsonType": "array", "items": {"bsonType": "string"}},
                        "improvementSuggestions": {"bsonType": "array", "items": {"bsonType": "string"}}
                    }
                },
                "llmAnalysis": {"bsonType": "string"},
                "llmModel": {"bsonType": "string"},
                "evaluationDate": {"bsonType": "date"}
            }
        }
    }
}



# Only create or update if needed
if "resumes" not in db.list_collection_names():
    db.create_collection("resumes", **resume_schema)
else:
    db.command({"collMod": "resumes", **resume_schema})


# Create job_descriptions collection if not present
if "job_descriptions" not in db.list_collection_names():
    db.create_collection("job_descriptions", **job_description_schema)
else:
    db.command({"collMod": "job_descriptions", **job_description_schema})


# Create evaluations collection if not present
if "evaluations" not in db.list_collection_names():
    db.create_collection("evaluations", **evaluations_schema)
else:
    db.command({"collMod": "evaluations", **evaluations_schema})

