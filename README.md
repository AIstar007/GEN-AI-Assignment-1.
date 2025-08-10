# Streamlit RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, FAISS, HuggingFace Embeddings, and LangChain.
It allows you to upload CSV or TXT files, process them into a FAISS vector store, and chat with the data.

## Features
- Upload CSV or TXT documents
- Robust CSV error handling
- HuggingFace embeddings for semantic search
- FAISS vector store for fast retrieval
- ChatGPT-like streaming responses
- Conversation history

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
streamlit run app.py
```

## File Structure
- `app.py` — Main Streamlit application
- `requirements.txt` — Python dependencies
- `README.md` — Project documentation
- `.gitignore` — Ignore unnecessary files in git

## Notes
- Ensure you have an internet connection for HuggingFace models.
- Supports both CSV and TXT file uploads.
