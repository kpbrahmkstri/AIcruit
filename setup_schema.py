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

# Only create or update if needed
if "resumes" not in db.list_collection_names():
    db.create_collection("resumes", **resume_schema)
else:
    db.command({"collMod": "resumes", **resume_schema})
