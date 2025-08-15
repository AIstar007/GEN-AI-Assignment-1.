import os
import io
import csv
import time
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
from typing import List
import re

import pandas as pd
import PyPDF2
from docx import Document as DocxDocument

# LangChain / embeddings / vectorstore imports (attempt)
EMBEDDINGS_OK = False
PINECONE_OK = False
HUGGINGFACE_EMBEDDINGS = None
PINECONE = None

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    HUGGINGFACE_EMBEDDINGS = HuggingFaceEmbeddings
    EMBEDDINGS_OK = True
except Exception as e:
    print(f"Embeddings import error: {e}")
    EMBEDDINGS_OK = False

try:
    import pinecone
    from langchain_community.vectorstores import Pinecone as PineconeStore
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENVIRONMENT") or "us-west1-gcp"
    if pinecone_api_key:
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
        PINECONE = PineconeStore
        PINECONE_OK = True
    else:
        PINECONE_OK = False
except Exception as e:
    print(f"Pinecone import error: {e}")
    PINECONE_OK = False

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain.memory import ConversationBufferWindowMemory

# Fix: Added missing import
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

PRIMARY = "#0b93f6"

HEADER_LOGO_URL = "https://cdn-icons-png.flaticon.com/512/4712/4712027.png"
ASSISTANT_LOGO_URL = "https://cdn-icons-png.flaticon.com/512/4712/4712027.png"

st.set_page_config(page_title="SAP Ariba RAG Chatbot", layout="wide")

# CSS Styles
st.markdown(f"""
    <style>
    .header-card {{ background-color: {PRIMARY}; padding:16px; border-radius:10px; color: white; display:flex; gap:16px; align-items:center; }}
    .chat-user {{
        background: linear-gradient(90deg, rgba(11,147,246,0.12), rgba(11,147,246,0.08));
        border-left: 4px solid {PRIMARY};
        padding:10px; border-radius:10px; margin:8px 0; display:flex; gap:8px; align-items:flex-start;
        color: #fff;
    }}
    .chat-assistant {{
        background:#f3f4f6; padding:10px; border-radius:10px; margin:8px 0; display:flex; gap:8px; align-items:flex-start;
        color: #222;
    }}
    .icon-left {{ width:28px; height:28px; margin-top:2px; }}
    .quiz-card {{ border:1px solid #eee; padding:14px; border-radius:10px; margin-bottom:12px; background: #fff; box-shadow: 0 2px 6px rgba(0,0,0,0.03); }}
    .question-badge {{ display:inline-block; background:{PRIMARY}; color:white; padding:6px 10px; border-radius:12px; font-weight:700; margin-right:10px; }}
    .correct {{ color: green; font-weight:700; }}
    .summary-card {{ border-left:6px solid {PRIMARY}; padding:12px; border-radius:8px; background:#fff; box-shadow:0 2px 6px rgba(0,0,0,0.03); margin-bottom:12px; }}
    .small-muted {{ color:#666; font-size:13px; }}
    .active-file {{ font-weight:700; color: {PRIMARY}; }}
    .chat-options-bar {{
        display: flex;
        align-items: center;
        background: #181818;
        border-radius: 8px;
        padding: 8px 12px;
        margin-bottom: 10px;
        gap: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }}
    .chat-options-bar .option-select {{
        background: #232323;
        color: #fff;
        border-radius: 6px;
        border: none;
        padding: 6px 10px;
        font-size: 15px;
        margin-right: 6px;
    }}
    .chat-options-bar .attach-btn {{
        background: #232323;
        color: #fff;
        border: none;
        border-radius: 6px;
        padding: 8px 10px;
        font-size: 15px;
        cursor: pointer;
        margin-right: 6px;
    }}
    .chat-options-bar .send-btn {{
        background: #0b93f6;
        color: #fff;
        border: none;
        border-radius: 6px;
        padding: 8px 16px;
        font-size: 15px;
        cursor: pointer;
        transition: background 0.2s;
        margin-right: 6px;
    }}
    .chat-options-bar .clear-btn {{
        background: #444;
        color: #fff;
        border: none;
        border-radius: 6px;
        padding: 8px 16px;
        font-size: 15px;
        cursor: pointer;
    }}
    </style>
""", unsafe_allow_html=True)

QNA_SYSTEM = """
You are SAP Ariba Expert Assistant. Use ONLY the provided context passages.
- If the topic or keyword appears in the context, return the relevant details from the context.
- Do NOT invent facts or use outside knowledge.
- If the answer is not present in the provided context, reply exactly: "I could not find the answer in the provided material."
- When you reference a fact, add a friendly source note like: (from filename.pdf)
Answer in a simple, customer-friendly tone.
"""

SUMMARY_SYSTEM = """
You are SAP Ariba Expert Assistant. Using ONLY the provided context, OUTPUT in Markdown:

### Summary
<3–6 sentences overview>

### Study Notes
- 5–7 concise bullets

### Key Definitions
- **Term**: short definition

If insufficient info, say: "The provided material does not contain enough information to summarize fully."
"""

QUIZ_SYSTEM = """
You are SAP Ariba Expert Assistant. Using ONLY the provided context, create EXACTLY 5 multiple-choice questions.
Output strictly in this format for each question:

Q1. Question text
A. option
B. option
C. option
D. option
Answer: X

Rules:
- Do not use outside knowledge.
- Make questions clear and relevant to the context.
"""

def load_csv_safely(uploaded_file) -> pd.DataFrame | None:
    try:
        sample = uploaded_file.read(200000).decode("utf-8", errors="ignore")
        uploaded_file.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        sep = dialect.delimiter
    except Exception:
        sep = ","
    try:
        df = pd.read_csv(uploaded_file, sep=sep, on_bad_lines="skip", engine="python", quotechar='"')
        return df
    except Exception:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, on_bad_lines="skip", engine="python")
            return df
        except Exception:
            return None

def extract_text_from_pdf(uploaded_file) -> str:
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
                text += f"\n--- Page {i+1} ---\n{page_text}\n"
            except Exception:
                continue
        return text
    except Exception:
        return ""

def extract_text_from_docx(uploaded_file) -> str:
    try:
        doc = DocxDocument(uploaded_file)
        out = []
        for p in doc.paragraphs:
            if p.text and p.text.strip():
                out.append(p.text)
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join([cell.text.strip() for cell in row.cells])
                if row_text:
                    out.append(row_text)
        return "\n".join(out)
    except Exception:
        return ""

class TFIDFWrapper:
    def __init__(self):
        self.texts = []
        self.metadatas = []
        self.vectorizer = None
        self.matrix = None

    def add_documents(self, documents: List[Document]):
        for d in documents:
            self.texts.append(d.page_content)
            self.metadatas.append(d.metadata if d.metadata else {})
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def from_documents(self, documents: List[Document]):
        self.texts = [d.page_content for d in documents]
        self.metadatas = [d.metadata if d.metadata else {} for d in documents]
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def get_relevant_documents(self, query: str, k: int = 3) -> List[Document]:
        if self.matrix is None or self.vectorizer is None or len(self.texts) == 0:
            return []
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        top_idxs = sims.argsort()[::-1][:k]
        results = []
        for i in top_idxs:
            results.append(Document(page_content=self.texts[i], metadata=self.metadatas[i]))
        return results

class EnhancedRAGChatbot:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.chain = None
        self.memory = None
        self.conversation_history = []
        self._init_components()

    def _init_components(self):
        if EMBEDDINGS_OK and PINECONE_OK:
            try:
                self.embeddings = HUGGINGFACE_EMBEDDINGS(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={"device": "cpu"}
                )
            except Exception:
                self.embeddings = None
        else:
            self.embeddings = None

        groq_key = os.getenv("GROQ_API_KEY") or GROQ_API_KEY
        if groq_key:
            try:
                model_name = st.session_state.get("selected_model", "gemma2-9b-it")
                temp = st.session_state.get("temperature", 0.1)
                self.llm = ChatGroq(groq_api_key=groq_key, model_name=model_name, temperature=temp)
            except Exception:
                self.llm = None
        else:
            self.llm = None

        try:
            self.memory = ConversationBufferWindowMemory(k=5, return_messages=True)
        except Exception:
            self.memory = None

        if self.llm:
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", QNA_SYSTEM + """

CONTEXT INFORMATION:
{context}

CONVERSATION HISTORY:
{chat_history}

Current Date: {current_date}"""),
                ("user", "{question}")
            ])
            output_parser = StrOutputParser()
            try:
                self.chain = prompt_template | self.llm | output_parser
            except Exception:
                self.chain = None

        self._load_default_documents()

    def _load_default_documents(self):
        docs = [
            Document(page_content="SAP Ariba Contract Management Process details...", metadata={"source": "Contract_Management_Guide"}),
            Document(page_content="SAP Ariba Sourcing Process details...", metadata={"source": "Sourcing_Process_Guide"}),
        ]
        if self.embeddings is not None and PINECONE_OK:
            try:
                index_name = "ariba-chatbot"
                if index_name not in pinecone.list_indexes():
                    pinecone.create_index(index_name, dimension=384)
                self.vectorstore = PINECONE.from_documents(docs, self.embeddings, index_name=index_name)
            except Exception:
                self.vectorstore = TFIDFWrapper()
                self.vectorstore.from_documents(docs)
        else:
            self.vectorstore = TFIDFWrapper()
            self.vectorstore.from_documents(docs)

    def add_documents(self, documents: List[Document]):
        if self.vectorstore is None:
            self.vectorstore = TFIDFWrapper()
        try:
            if PINECONE_OK and hasattr(self.vectorstore, "add_documents") and not isinstance(self.vectorstore, TFIDFWrapper):
                self.vectorstore.add_documents(documents)
            elif isinstance(self.vectorstore, TFIDFWrapper):
                self.vectorstore.add_documents(documents)
            else:
                self.vectorstore = TFIDFWrapper()
                self.vectorstore.from_documents(documents)
        except Exception:
            self.vectorstore = TFIDFWrapper()
            self.vectorstore.from_documents(documents)

    def chat(self, question: str) -> str:
        if self.vectorstore is None:
            return "No documents indexed. Upload documents first."
        try:
            if PINECONE_OK and hasattr(self.vectorstore, "as_retriever") and not isinstance(self.vectorstore, TFIDFWrapper):
                retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
                docs = retriever.get_relevant_documents(question)
            elif isinstance(self.vectorstore, TFIDFWrapper):
                docs = self.vectorstore.get_relevant_documents(question, k=3)
            else:
                if hasattr(self.vectorstore, "get_relevant_documents"):
                    docs = self.vectorstore.get_relevant_documents(question)
                else:
                    docs = []
        except Exception as e:
            return f"Retriever error: {e}"

        context_text = "\n\n".join([d.page_content for d in docs])
        chat_history = "\n".join(self.conversation_history[-10:])
        current_date = datetime.now().strftime("%Y-%m-%d")

        if self.chain is None:
            if context_text.strip():
                answer = f"Context found from documents:\n\n{context_text[:1000]}...\n\n(LLM unavailable — set GROQ_API_KEY to enable LLM answers.)"
            else:
                answer = "No contextual documents found and LLM not available."
        else:
            try:
                answer = self.chain.invoke({
                    "context": context_text,
                    "chat_history": chat_history,
                    "question": question,
                    "current_date": current_date
                })
            except Exception as e:
                answer = f"LLM/chain error: {e}"

        self.conversation_history.append(f"User: {question}")
        self.conversation_history.append(f"AI: {answer}")
        return answer

# Initialize session state
if "chatbot" not in st.session_state:
    st.session_state.chatbot = EnhancedRAGChatbot()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gemma2-9b-it"
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.1

# Initialize Summary & Quiz Feature State
if "summary_output" not in st.session_state:
    st.session_state.summary_output = None
if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = None
if "quiz_index" not in st.session_state:
    st.session_state.quiz_index = 0
if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = 0
if "quiz_done" not in st.session_state:
    st.session_state.quiz_done = False
if "quiz_feedback" not in st.session_state:
    st.session_state.quiz_feedback = None

# Main app
st.title("SAP Ariba RAG Chatbot")
st.write("Upload documents and ask questions about SAP Ariba!")

# File upload
uploaded_files = st.file_uploader(
    "Choose files",
    type=["pdf", "docx", "txt", "csv", "xls", "xlsx"],
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("Process Files"):
        docs = []
        for f in uploaded_files:
            name = f.name.lower()
            content = ""
            if name.endswith(".pdf"):
                content = extract_text_from_pdf(f)
            elif name.endswith(".docx"):
                content = extract_text_from_docx(f)
            elif name.endswith(".txt"):
                try:
                    f.seek(0)
                    content = f.read().decode("utf-8", errors="ignore")
                except Exception:
                    content = ""
            elif name.endswith(".csv"):
                f.seek(0)
                df = load_csv_safely(f)
                if df is not None:
                    content = df.to_string(index=False)
            elif name.endswith((".xls", ".xlsx")):
                try:
                    f.seek(0)
                    df = pd.read_excel(f, engine="openpyxl")
                    content = df.to_string(index=False)
                except Exception:
                    content = ""
            else:
                content = ""
            if content and content.strip():
                docs.append(Document(page_content=content, metadata={"source": f.name}))

        if docs:
            st.session_state.chatbot.add_documents(docs)
            st.success(f"Indexed {len(docs)} uploaded files.")
        else:
            st.warning("No content extracted from uploaded files.")

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Ask me anything about SAP Ariba"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chatbot.chat(prompt)
            st.write(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
