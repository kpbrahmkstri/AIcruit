import streamlit as st
import requests
import pandas as pd
import altair as alt
from streamlit_option_menu import option_menu
import io
import pdfplumber
from docx import Document

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="AIcruit", layout="wide")

# Sidebar Navigation
with st.sidebar:
    st.image("AIcruit_logo.png", width=250)
    selected = option_menu(
        menu_title=None,
        options=["📄 Applicant Portal", "📄 Recruiter Portal", "📊 Recruiter Dashboard"],
        icons=[None] * 3,
        default_index=0,
        orientation="vertical",
        styles={
            "container": {"padding": "0!important", "background-color": "#f8f9fa"},
            "icon": {"color": "black", "font-size": "0px"},
            "nav-link": {
                "font-size": "14px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#e0f0ff"
            },
            "nav-link-selected": {
                "background-color": "#3399ff",
                "color": "white"
            },
        }
    )

# Shared: Fetch Job Titles
def fetch_job_titles():
    try:
        response = requests.get(f"{API_BASE}/job-titles/")
        if response.status_code == 200:
            return ["Select Job Title"] + response.json().get("titles", [])
        else:
            st.error("❌ Could not load job titles.")
    except Exception as e:
        st.error(f"Exception while fetching job titles: {e}")
    return ["Select Job Title"]

# === APPLICANT PORTAL ===
if selected == "📄 Applicant Portal":
    st.title("📄 Upload Resume")
    job_titles = fetch_job_titles()
    selected_job = st.selectbox("Select Job Title", job_titles)
    uploaded_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])

    if uploaded_file and selected_job != "Select Job Title":
        if st.button("Submit Resume"):
            with st.spinner("Uploading and evaluating resume..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    data = {"job_title": selected_job}
                    response = requests.post(f"{API_BASE}/upload-resume/", data=data, files=files)
                    if response.status_code == 200:
                        st.success("✅ Resume uploaded and evaluated successfully!")
                        st.json(response.json())
                    else:
                        st.error(f"❌ Error {response.status_code}: {response.text}")
                except Exception as e:
                    st.error(f"Exception occurred: {e}")
    elif uploaded_file:
        st.warning("⚠️ Please select a valid job title before uploading.")
    else:
        st.info("📥 Upload your resume and select a job title to proceed.")

# === RECRUITER PORTAL ===
# === RECRUITER PORTAL ===
elif selected == "📄 Recruiter Portal":
    st.title("📄 Upload Job Description")
    job_title = st.text_input("Job Title", placeholder="e.g., Data Scientist")
    jd_file = st.file_uploader("Upload JD File (PDF/DOCX)", type=["pdf", "docx"])
    jd_text = ""
    jd_bytes = None

    if jd_file:
        try:
            jd_bytes = jd_file.read()  # Read once and reuse
            if jd_file.name.endswith(".pdf"):
                with pdfplumber.open(io.BytesIO(jd_bytes)) as pdf:
                    jd_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            elif jd_file.name.endswith(".docx"):
                doc = Document(io.BytesIO(jd_bytes))
                jd_text = "\n".join([para.text for para in doc.paragraphs])
            else:
                st.warning("Unsupported file type.")
        except Exception as e:
            st.error(f"Failed to read JD file: {e}")

    jd_input = st.text_area("Job Description", value=jd_text, height=250)

    if st.button("Submit JD"):
        if not job_title.strip():
            st.warning("Please enter a job title.")
        elif not jd_input.strip():
            st.warning("Please enter or upload a job description.")
        else:
            with st.spinner("Uploading JD and parsing via GPT..."):
                try:
                    files = {"file": (jd_file.name, io.BytesIO(jd_bytes), jd_file.type)} if jd_bytes else {}
                    data = {"job_title": job_title, "job_desc_input": jd_input}
                    response = requests.post(f"{API_BASE}/upload-jd/", data=data, files=files)
                    if response.status_code == 200:
                        st.success("✅ Job description uploaded and parsed!")
                        st.json(response.json())
                    else:
                        st.error(f"❌ Error {response.status_code}: {response.text}")
                except Exception as e:
                    st.error(f"Exception occurred: {e}")

# === RECRUITER DASHBOARD ===
elif selected == "📊 Recruiter Dashboard":
    st.title("📊 Top Resume Matching Dashboard")

    job_titles = fetch_job_titles()
    selected_title = st.selectbox("Select Job Title", job_titles)

    if selected_title != "Select Job Title":
        with st.spinner("Fetching resume matches..."):
            try:
                response = requests.get(f"{API_BASE}/dashboard/{selected_title}")
                data = response.json()

                if not data.get("rows"):
                    st.info("📭 No applications yet for this job title.")
                else:
                    rows = data["rows"]
                    scatter = data["scatter"]

                    st.subheader("🏆 Top Matching Resumes")
                    df = pd.DataFrame(rows)
                    df["Rank"] = df["Match Overall"].rank(ascending=False).astype(int)
                    df = df.sort_values("Rank")

                    st.write(
                        df[[
                            "Rank", "First Name", "Last Name", "Email", "Resume",
                            "Match Overall", "Skills Match", "Experience Match",
                            "Education Match", "Certification Match", "Keyword Match",
                            "Missing Skills", "Strengths", "Weaknesses", "Upload Time"
                        ]].to_html(escape=False, index=False),
                        unsafe_allow_html=True
                    )

                    if scatter:
                        scatter_df = pd.DataFrame(scatter)

                        # Filter out rows with missing or invalid data
                        scatter_df = scatter_df.dropna(subset=["Skills Match", "Experience Match"])

                        if not scatter_df.empty:
                            st.subheader("📈 Skills vs Experience Plot")
                            chart = alt.Chart(scatter_df).mark_circle(size=130).encode(
                                x=alt.X("Skills Match:Q", scale=alt.Scale(domain=[0, 110])),
                                y=alt.Y("Experience Match:Q", scale=alt.Scale(domain=[0, 110])),
                                color="Type:N",
                                tooltip=["Name", "Skills Match", "Experience Match"]
                            ).interactive()
                            st.altair_chart(chart, use_container_width=True)
                        else:
                            st.info("📭 No valid resume data to plot.")
            except Exception as e:
                st.error(f"❌ Failed to fetch data: {e}")
    else:
        st.info("Please select a Job Title.")
