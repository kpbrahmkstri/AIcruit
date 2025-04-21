import streamlit as st
import pandas as pd

# Set page configuration
st.set_page_config(page_title="AIcruit", page_icon=":guardsman:", layout="wide")

# --- Top Header with Logo ---
col1, col2 = st.columns([0.1, 0.9])
with col1:
    st.image("AIcruit_logo.png", width=150)  # Replace with actual logo path
with col2:
    st.title("AIcruit")

st.markdown("---")

# --- Tabs Layout ---
tab1, tab2 = st.tabs(["📄 Resume Upload", "📊 Top Resume Matching"])

# --- Tab 1: Resume Upload ---
with tab1:
    st.header("Upload Resume")
    st.write("Upload a resume (PDF or DOCX) to analyze and score against job descriptions.")

    uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])
    if uploaded_file:
        st.success("Resume uploaded successfully!")
        # Placeholder for future parsing logic

# --- Tab 2: Job Description & Resume Matching ---
with tab2:
    st.header("Job Description Matching")
    st.write("Select a job description and view top matching resumes.")

    # Job description selector
    job_desc_options = ['Software Engineer', 'Data Scientist', 'Web Developer']
    selected_job_desc = st.selectbox("Select Job Description Template", job_desc_options)

    if st.button("Find Top Resumes"):
        st.info(f"Finding top resumes for: {selected_job_desc}...")

        # Placeholder scoring result
        data = {
            "Candidate": ["John Doe", "Jane Smith", "Alice Johnson", "Bob Brown"],
            "Score": [95, 88, 92, 85],
            "Match %": [90, 80, 85, 75]
        }
        df = pd.DataFrame(data)

        st.subheader("Top Matching Resumes:")
        st.dataframe(df)

