# AICruit

A GenAI based solution to identify the top talent.

## Features

- Resume upload and parsing
- Job description upload and parsing
- Ranking resumes alongwith strengths and weaknesses
- Analytics dashboard

## Setup

```bash
# Clone the repo
git clone https://github.com/kpbrahmkstri/AIcruit.git
cd AIcruit

# (Optional) Create a virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

#Create .env file in the project root folder to read the API KEYS
MONGO_URI=<value>
OPENAI_API_KEY=<value>

#Run the app using the commands

#1. First run below to start backend:
uvicorn main:app --reload

#2. Now in another terminal, run frontend:
streamlit run .\frontend.py
