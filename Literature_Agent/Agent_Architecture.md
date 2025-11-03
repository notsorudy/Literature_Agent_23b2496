# AI Agent Architecture

## Components
- Streamlit UI (user interaction)
- arXiv API (paper fetching)
- PDF extraction (PyPDF2)
- Gemini API (summarization via Google Generative AI)
- Retrieval system (SentenceTransformer + FAISS for context retrieval)

## Interaction Flow
1. User searches for arXiv papers or uploads PDF
2. Text/abstract is indexed for retrieval
3. RAG system retrieves relevant contexts
4. Summarization prompt sent to Gemini with context
5. Summary displayed, with download/export features

## Models Used
- SentenceTransformer: all-MiniLM-L6-v2 (embeddings for retrieval)
- Google Gemini (LLM for summarization)

## Design Choices
- RAG enhances factual accuracy and detail
- Streamlit chosen for rapid prototyping and accessible UI
- Gemini LLM for high-quality summaries
