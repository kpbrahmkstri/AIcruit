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
                "parsed": {
                    "bsonType": "object",
                    "required": ["work_experience", "education", "skills", "achievements"],
                    "properties": {
                        "work_experience": {"bsonType": "string"},
                        "education": {"bsonType": "string"},
                        "skills": {"bsonType": "string"},
                        "achievements": {"bsonType": "string"}
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
                "job_title":     {"bsonType": "string"},
                "upload_time":   {"bsonType": "date"},
                "file_type":     {"bsonType": "string"},
                "filename":      {"bsonType": "string"},
                "file_data":     {"bsonType": ["binData", "null"]},
                "description":   {"bsonType": "string"},

                # Optional GPT-parsed structure
                "parsed": {
                    "bsonType": "object",
                    "required": ["responsibilities", "required_skills", "qualifications"],
                    "properties": {
                        "responsibilities": {"bsonType": "array", "items": {"bsonType": "string"}},
                        "required_skills":  {"bsonType": "array", "items": {"bsonType": "string"}},
                        "qualifications":   {"bsonType": "array", "items": {"bsonType": "string"}}
                    }
                }
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
