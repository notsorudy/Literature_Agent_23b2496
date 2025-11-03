import streamlit as st
import arxiv
import google.generativeai as genai
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime
import tempfile
from PyPDF2 import PdfReader

# Additional imports for RAG retrieval
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Fix arXiv HTTP 301 error: force HTTPS endpoint
arxiv.Search.base_url = "https://export.arxiv.org/api/query"

# --- Load API Key ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY not found in environment variables.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# --- Streamlit Page Setup ---
st.set_page_config(page_title="AI Literature Summarizer", page_icon="📚", layout="wide")
st.title("📚 AI Literature Summarizer with Retrieval")
st.markdown("Summarize **arXiv papers** or **your own PDFs** using Google Gemini enhanced by retrieval.")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Options")
    mode = st.radio("Choose mode:", ["🔍 Search arXiv", "📄 Upload PDF"])
    model_choice = st.selectbox(
        "🤖 Choose Gemini Model:",
        ["models/gemini-2.5-flash", "models/gemini-2.5-pro"],
        index=0
    )

# --- Helper Functions ---

def fetch_papers(query, n=3):
    """Fetch top N relevant papers from arXiv."""
    search = arxiv.Search(query=query, max_results=n, sort_by=arxiv.SortCriterion.Relevance)
    papers = []
    for result in search.results():
        papers.append({
            "title": result.title,
            "authors": [a.name for a in result.authors],
            "year": result.published.year,
            "abstract": result.summary,
            "url": result.entry_id
        })
    return papers

def extract_text_from_pdf(uploaded_file):
    """Extract text from a PDF file."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        reader = PdfReader(tmp_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        return f"⚠️ PDF extraction failed: {e}"

# Preload sentence transformer for embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2") # lightweight, fast
index = None
corpus_texts = []
corpus_embeddings = None

def build_faiss_index(texts):
    global index, corpus_texts, corpus_embeddings
    corpus_texts = texts
    corpus_embeddings = embedder.encode(texts, convert_to_numpy=True)
    dim = corpus_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product -> cosine sim after normalize
    faiss.normalize_L2(corpus_embeddings)
    index.add(corpus_embeddings)

def retrieve_context(query, top_k=5):
    """Retrieve top_k similar texts from indexed corpus."""
    if index is None or len(corpus_texts) == 0:
        return ""
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    results = [corpus_texts[i] for i in I[0]]
    return "\n\n".join(results)

def summarize_text(title, text, model_choice, use_retrieval=False):
    """Summarize text (paper abstract or extracted PDF text) with optional retrieval."""
    context = ""
    if use_retrieval:
        context = retrieve_context(title + " " + text)

    prompt = f"""
Please provide a detailed and comprehensive summary of the following academic paper or section.

Title: {title}

Context: {context}

Text: {text[:10000]}  # Increase token length if possible for Gemini

Your detailed summary should include:
- 🎯 Clear statement of the research problem and objectives
- 🧪 Description of methodologies and experiments conducted
- 🔑 Results and findings with quantitative or qualitative detail
- 📊 Discussion of implications and significance of the work
- ⚠️ Any stated limitations and areas for future work

Format the answer in clear paragraphs or detailed bullet points.
"""

    try:
        model = genai.GenerativeModel(model_choice)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Summarization failed: {e}"

# --- Build initial corpus for retrieval from arXiv abstracts (can be expanded) ---
@st.cache_data(show_spinner=True)
def build_initial_corpus():
    sample_papers = fetch_papers("machine learning", n=100)
    texts = [paper["abstract"] for paper in sample_papers]
    build_faiss_index(texts)
    return len(texts)

corpus_size = build_initial_corpus()
st.sidebar.caption(f"Indexed {corpus_size} arXiv abstracts for retrieval")

# --- Main app logic ---

if mode == "🔍 Search arXiv":
    query = st.text_input("🔍 Enter your research topic:", "AI-based materials discovery")
    n_papers = st.slider("📄 Number of papers:", 1, 10, 3)
    use_rag = st.checkbox("Use retrieval-augmented summarization", value=True)
    summarize_btn = st.button("🚀 Fetch & Summarize")

    if summarize_btn:
        with st.spinner("🔎 Fetching papers..."):
            papers = fetch_papers(query, n_papers)
        st.success(f"✅ Found {len(papers)} papers related to '{query}'.")

        summaries = []
        for i, paper in enumerate(papers, start=1):
            st.markdown(f"### 📄 Paper {i}: {paper['title']}")
            st.markdown(f"**👥 Authors:** {', '.join(paper['authors'])}")
            st.markdown(f"**📅 Year:** {paper['year']}")
            st.markdown(f"[🔗 View on arXiv]({paper['url']})")

            with st.spinner(f"🧠 Summarizing paper {i}..."):
                summary = summarize_text(paper["title"], paper["abstract"], model_choice, use_rag)
                st.write(summary)

            paper["summary"] = summary
            summaries.append(paper)
            st.divider()

        # Save and download
        if summaries:
            df = pd.DataFrame(summaries)
            filename = f"summaries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            st.download_button("💾 Download Summaries (CSV)", df.to_csv(index=False), file_name=filename)
            st.balloons()

else:  # Upload PDF mode
    uploaded_files = st.file_uploader("📤 Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        for i, uploaded_file in enumerate(uploaded_files, start=1):
            st.markdown(f"### 📘 File {i}: {uploaded_file.name}")

            with st.spinner("📄 Extracting text from PDF..."):
                text = extract_text_from_pdf(uploaded_file)

            if text.startswith("⚠️"):
                st.error(text)
                continue
            
            use_rag = st.checkbox("Use retrieval-augmented summarization", value=True, key=f"rag_pdf_{i}")

            with st.spinner("🧠 Summarizing with Gemini..."):
                summary = summarize_text(uploaded_file.name, text, model_choice, use_rag)
            
            st.write(summary)
            st.divider()

        st.success("✅ All uploaded PDFs summarized!")
