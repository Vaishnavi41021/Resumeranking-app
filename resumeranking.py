import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64
import os
import time
import matplotlib.pyplot as plt

# Function to set the background image
def set_background(image_file):
    """Sets a semi-transparent background image in Streamlit."""
    if not os.path.exists(image_file):
        st.error("Background image not found. Check the file path.")
        return
    
    with open(image_file, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()
    
    background_css = f"""
    <style>
    .stApp {{
        background: url("data:image/png;base64,{encoded_string}") no-repeat center center fixed;
        background-size: cover;
    }}
    
    .stApp::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(255, 255, 255, 0.3);  /* Reduced transparency */
        z-index: -1;
    }}
    
    .block-container {{
        background: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        margin-top: 80px;
    }}
    </style>
    """
    st.markdown(background_css, unsafe_allow_html=True)

# Set background image (update the correct path)
image_path = "assets/backgroundAicte.png"  # Ensure the correct path
set_background(image_path)

# Function to extract text from PDFs
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    vectorizer = TfidfVectorizer()
    documents = [job_description] + resumes
    tfidf_matrix = vectorizer.fit_transform(documents)
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    ranked_resumes = sorted(enumerate(similarity_scores), key=lambda x: x[1], reverse=True)
    return ranked_resumes, similarity_scores

# Page Title
st.markdown(
    "<h1 style='text-align: center; color: var(--text-color, black);'>Resume Ranking System</h1>",
    unsafe_allow_html=True,
)

# Dark Mode Toggle
dark_mode = st.checkbox("🌙 Dark Mode")
if dark_mode:
    st.markdown("""
        <style>
        body, .stApp { background-color: #121212 !important; color: white !important; }
        .block-container { background: rgba(50, 50, 50, 0.9) !important; }
        .stTextInput, .stTextArea, .stFileUploader { color: white !important; }
        .stProgress { background-color: #444 !important; }
        </style>
    """, unsafe_allow_html=True)

# Upload PDFs
uploaded_files = st.file_uploader("Upload resumes (PDF)", accept_multiple_files=True)
job_desc = st.text_area("Enter Job Description")

# Show uploaded files
if uploaded_files:
    st.subheader("Uploaded Files")
    for file in uploaded_files:
        st.write(f"📄 {file.name} ({file.size / 1024:.2f} KB)")

# Rank resumes
if st.button("Rank Resumes"):
    if uploaded_files and job_desc:
        with st.spinner("Ranking resumes, please wait..."):
            time.sleep(2)
            resumes_text = [extract_text_from_pdf(file) for file in uploaded_files]
            rankings, scores = rank_resumes(job_desc, resumes_text)
            
            st.subheader("Ranked Resumes")
            result_data = []
            for i, (index, score) in enumerate(rankings):
                st.write(f"Rank {i+1}: {uploaded_files[index].name} (Score: {score:.2f})")
                st.progress(int(score * 100))
                result_data.append([uploaded_files[index].name, score])
            
            df = pd.DataFrame(result_data, columns=["Resume Name", "Score"])
            
            # Bar Chart Visualization
            st.subheader("Resume Ranking Visualization")
            fig, ax = plt.subplots()
            ax.barh([uploaded_files[i].name for i, _ in rankings], scores, color=['blue', 'purple', 'orange', 'yellow'])
            ax.set_xlabel("Similarity Score")
            ax.set_title("Resume Ranking Results")
            ax.invert_yaxis()  # Highest rank at top
            st.pyplot(fig)
            
            # Download button for ranked resumes
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results", data=csv, file_name="ranked_resumes.csv", mime="text/csv")
    else:
        st.warning("Please upload resumes and enter a job description.")
