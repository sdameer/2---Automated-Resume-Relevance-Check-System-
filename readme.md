# 2 - Automated Resume Relevance Check System 

This project is a **Streamlit web application** that automates the process of comparing multiple candidate resumes with a job description and provides a **match score and verdict**. It is ideal for recruiters, HR teams, and hiring managers to quickly shortlist candidates for a position.

---

## **Features**

- Upload a **Job Description (PDF)**.  
- Upload **multiple resumes (PDFs)** at once.  
- Automatically extracts text from PDFs.  
- Uses **Chroma DB** to store document chunks and embeddings for efficient processing.  
- Compares resumes with the job description using **LLM (ChatGroq)**.  
- Returns a **JSON score (0–100)** and **verdict** (`high`, `medium`, `low`).  
- Displays results in a clean Streamlit interface.  
- Works for **10–50+ resumes** efficiently.

---

## **How It Works**

1. **Upload PDFs**:  
   - The recruiter uploads a job description PDF and one or more candidate resumes.  

2. **Text Extraction & Chunking**:  
   - The PDF files are processed using `PyPDFLoader`.  
   - Text is split into chunks using `RecursiveCharacterTextSplitter` to handle large documents.

3. **Embedding & Storage (Chroma DB)**:  
   - Each chunk is converted into embeddings using `HuggingFaceEmbeddings`.  
   - Embeddings are stored in **Chroma DB**, which allows efficient retrieval and comparison.

4. **Candidate Comparison**:  
   - Each candidate’s text is combined and sent to **ChatGroq LLM** along with the job description.  
   - The LLM returns a **JSON object** with `score` and `verdict` indicating the suitability of the candidate.

5. **Results Display**:  
   - Streamlit displays a **table of candidate names, scores, and verdicts**.

---

## **Tech Stack**

- **Python**  
- **Streamlit** for UI  
- **LangChain** for document loading and LLM integration  
- **Chroma DB** for embeddings storage  
- **HuggingFace Transformers** for embeddings  
- **ChatGroq** for resume-job comparison using LLM  

