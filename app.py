import os
import warnings
import re
import json
import tempfile
import shutil
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
DB_DIR = os.path.join(tempfile.gettempdir(), "chroma_db")
os.makedirs(DB_DIR, exist_ok=True)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"


embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
llm = ChatGroq(model=LLM_MODEL,)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""]
)

# ---------------- HELPERS ----------------
def sanitize_name(name: str) -> str:
    name = name.replace(" ", "_")
    name = re.sub(r"[^a-zA-Z0-9._-]", "", name)
    name = re.sub(r"^[^a-zA-Z0-9]+", "", name)
    name = re.sub(r"[^a-zA-Z0-9]+$", "", name)
    return name[:512]

def add_pdf_to_collection(pdf_path, collection_name):
    collection_name = sanitize_name(collection_name)
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    if not docs:
        print(f"⚠️ PDF {pdf_path} has no readable text!")
        return
    chunks = splitter.split_documents(docs)
    if not chunks:
        print(f"⚠️ PDF {pdf_path} split into 0 chunks!")
        return
    collection = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embed,
        collection_name=collection_name
    )
    collection.add_documents(chunks)
    print(f"✅ Added {pdf_path} to collection: {collection_name}")

def clean_llm_json(output: str) -> dict:
    output = re.sub(r"```json|```", "", output, flags=re.IGNORECASE).strip()
    match = re.search(r"\{.*\}", output, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass
    return {"raw": output}

def compare_candidate(candidate_name: str, job_collection_name: str = "job_description") -> dict:
    job_collection = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embed,
        collection_name=job_collection_name
    )
    candidate_collection = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embed,
        collection_name=candidate_name
    )

    job_docs = job_collection.get(include=["documents"])["documents"]
    candidate_docs = candidate_collection.get(include=["documents"])["documents"]

    job_text = "\n".join(job_docs)
    candidate_text = "\n".join(candidate_docs)

    prompt = f"""
You are a recruitment assistant. Compare the following job description with the candidate profile.

Job Description:
{job_text[:1500]}

Candidate Resume:
{candidate_text[:1500]}

Return a JSON ONLY in this format:
{{
    "score": "<0-100>",
    "verdict": "high / medium / low"
}}
"""
    messages = [
        SystemMessage(content="You are an expert recruitment assistant."),
        HumanMessage(content=prompt)
    ]

    raw_result = llm.invoke(messages)
    result = clean_llm_json(raw_result.content)
    return result

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Resume Matcher", layout="wide")
st.title("📊 Resume vs Job Description Matcher")

jd_file = st.file_uploader("📄 Upload Job Description (PDF)", type=["pdf"])
resume_files = st.file_uploader("📂 Upload Resumes (PDFs)", type=["pdf"], accept_multiple_files=True)

if jd_file and resume_files and st.button("🚀 Run Matching"):

    # ---------------- Clear previous collections safely ----------------
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR, ignore_errors=True)
    os.makedirs(DB_DIR, exist_ok=True)

    # ---------------- Process Job Description ----------------
    st.write("📌 Processing Job Description...")
    jd_temp = os.path.join(DB_DIR, "job_description.pdf")
    with open(jd_temp, "wb") as f:
        f.write(jd_file.read())
    add_pdf_to_collection(jd_temp, "job_description")

    # ---------------- Process Resumes ----------------
    st.write("📌 Processing Resumes...")
    candidate_names = []
    for file in resume_files:
        cand_name = sanitize_name(os.path.splitext(file.name)[0])
        cand_temp = os.path.join(DB_DIR, file.name)
        with open(cand_temp, "wb") as f:
            f.write(file.read())
        add_pdf_to_collection(cand_temp, cand_name)
        candidate_names.append(cand_name)

    # ---------------- Compare Candidates ----------------
    st.write("🤖 Running candidate-job comparisons...")
    results = {}
    for cand in candidate_names:
        try:
            results[cand] = compare_candidate(cand)
            st.success(f"✅ Finished comparison for `{cand}` → {results[cand]}")
        except Exception as e:
            st.error(f"⚠️ Error comparing {cand}: {e}")
            results[cand] = {"error": str(e)}

    # ---------------- Save and Display Results ----------------
    with open("candidate_scores.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    st.success("📂 Results saved to `candidate_scores.json`")
    st.subheader("📌 Final Results")
    for name, res in results.items():
        col1, col2, col3 = st.columns([3, 1, 2])
        with col1: st.markdown(f"**{name}**")
        with col2: st.markdown(f"🎯 {res.get('score', 'N/A')}")
        with col3: st.markdown(f"✅ {res.get('verdict', 'N/A')}")
