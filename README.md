# 🚀 RAG Chatbot  

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Chatbot-ff4b4b?logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-RAG-orange)
![Pinecone](https://img.shields.io/badge/Pinecone-VectorDB-green)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Embeddings-yellow?logo=huggingface)

A **Retrieval-Augmented Generation (RAG) chatbot** built with **Streamlit, Pinecone, HuggingFace Embeddings, and LangChain**.  
This chatbot lets you **upload your own data** (CSV, TXT, PDF, DOCX, Excel) and interact with it conversationally, just like ChatGPT.  

---

## ✨ Features  
- 📂 Upload and process **CSV, TXT, PDF, DOCX, XLSX** documents  
- 🔍 **Semantic search** powered by HuggingFace embeddings  
- 📡 **Vector storage** with Pinecone (or TF-IDF fallback if unavailable)  
- 💬 ChatGPT-like **streaming responses**  
- 📝 Extra tools: **Summarization** & **Quiz generation** from uploaded docs  

---

## ⚙️ Installation  

1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/GEN-AI-Assignment-1..git
   cd GEN-AI-Assignment-1.
   ```

2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:  
   - Create a `.env` file in the root folder  
   - Add your API keys:  
     ```ini
     GROQ_API_KEY=your_groq_api_key
     PINECONE_API_KEY=your_pinecone_api_key
     PINECONE_ENVIRONMENT=us-west1-gcp
     ```

---

## ▶️ Usage  

Run the app with:  
```bash
streamlit run app.py
```

Then open your browser at **http://localhost:8501** 🎉  

---

## 📂 File Structure  

```
.
├── app.py             # Main Streamlit application
├── requirements.txt   # Python dependencies
├── README.md          # Project documentation
├── .gitignore         # Ignore unnecessary files in git
└── .env.example       # Example environment variables
```

---

## 📝 Notes  

- Ensure you have a stable **internet connection** (for HuggingFace & Groq models).  
- If **Pinecone** is not configured, the app automatically falls back to **TF-IDF search**.  
- Works with both **structured (CSV, Excel)** and **unstructured (TXT, PDF, DOCX)** documents.  

---

## 💡 Example Use Cases  

- Upload your **training manuals** → ask questions instantly  
- Feed in **contracts/policies** → get quick summaries  
- Use for **learning** (quiz generation, note-taking)  
- Or just as a **personal ChatGPT for your own files** 📑🤖  

---

🔥 With this chatbot, your documents become interactive knowledge bases!  
