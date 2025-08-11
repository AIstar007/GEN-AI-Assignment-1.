import os
import io
import csv
import time
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
from typing import List

import pandas as pd
import PyPDF2
from docx import Document as DocxDocument

# LangChain / embeddings / vectorstore imports (Qdrant)
EMBEDDINGS_OK = False
QDRANT_OK = False
HUGGINGFACE_EMBEDDINGS = None
QDRANT = None

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    HUGGINGFACE_EMBEDDINGS = HuggingFaceEmbeddings
    EMBEDDINGS_OK = True
except Exception:
    EMBEDDINGS_OK = False

try:
    from langchain_community.vectorstores import Qdrant as QdrantStore
    import qdrant_client
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY", None)
    QDRANT_CLIENT = qdrant_client.QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    QDRANT = QdrantStore
    QDRANT_OK = True
except Exception:
    QDRANT_OK = False

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain.memory import ConversationBufferWindowMemory

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="SAP Ariba RAG Chatbot", layout="wide")
st.markdown("""
    <style>
    .main {
        background-color: #f7f9fa;
    }
    .chat-card {
        border-radius: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        margin-bottom: 12px;
        padding: 16px;
        background: #fff;
        max-width: 90%;
    }
    .user-msg {
        background: linear-gradient(90deg, #0b93f6 60%, #005bea 100%);
        color: #fff;
        text-align: right;
    }
    .assistant-msg {
        background: #f1f3f4;
        color: #222;
        text-align: left;
    }
    .avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        margin-right: 8px;
        vertical-align: middle;
    }
    .footer {
        text-align: center;
        color: #888;
        font-size: 13px;
        margin-top: 32px;
    }
    </style>
""", unsafe_allow_html=True)

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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
        if EMBEDDINGS_OK and QDRANT_OK:
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
                ("system", """You are SAP Ariba Expert Assistant...

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
        if self.embeddings is not None and QDRANT_OK:
            try:
                collection_name = "ariba-chatbot"
                self.vectorstore = QDRANT.from_documents(
                    docs,
                    self.embeddings,
                    url=os.getenv("QDRANT_URL", "http://localhost:6333"),
                    api_key=os.getenv("QDRANT_API_KEY", None),
                    collection_name=collection_name
                )
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
            if QDRANT_OK and hasattr(self.vectorstore, "add_documents") and not isinstance(self.vectorstore, TFIDFWrapper):
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
            if QDRANT_OK and hasattr(self.vectorstore, "as_retriever") and not isinstance(self.vectorstore, TFIDFWrapper):
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

def process_uploaded_files(uploaded_files):
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

def stream_assistant_text(text: str, placeholder: st.delta_generator.DeltaGenerator):
    words = text.split()
    out = ""
    for w in words:
        out += w + " "
        html = out.replace("\n", "<br>")
        placeholder.markdown(f"<div style='text-align:left; background:#f1f3f4; color:#111; padding:10px 12px; border-radius:12px; margin:6px 0;'>{html}</div>", unsafe_allow_html=True)
        time.sleep(0.02)
    placeholder.markdown(f"<div style='text-align:left; background:#f1f3f4; color:#111; padding:10px 12px; border-radius:12px; margin:6px 0;'>{html}</div>", unsafe_allow_html=True)

def on_send():
    text = st.session_state.user_input.strip()
    if not text:
        st.warning("Please enter a message.")
        return
    st.session_state.messages.append({
        "role": "user",
        "content": text,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })
    try:
        st.session_state.chatbot.llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=st.session_state.get("selected_model", "gemma2-9b-it"),
            temperature=st.session_state.get("temperature", 0.1)
        )
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are SAP Ariba Expert Assistant...

CONTEXT INFORMATION:
{context}

CONVERSATION HISTORY:
{chat_history}

Current Date: {current_date}"""),
            ("user", "{question}")
        ])
        st.session_state.chatbot.chain = prompt_template | st.session_state.chatbot.llm | StrOutputParser()
    except Exception:
        pass

    with st.spinner("Generating answer..."):
        resp = st.session_state.chatbot.chat(text)

    st.session_state.messages.append({
        "role": "assistant",
        "content": "",
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })
    st.session_state.user_input = ""

    placeholder = st.empty()
    stream_assistant_text(resp, placeholder)

    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        st.session_state.messages[-1]["content"] = resp

def on_clear():
    st.session_state.messages = []
    st.session_state.chatbot.conversation_history = []

with st.sidebar:
    st.title("🔧 Settings & Uploads")
    uploaded_files = st.file_uploader("Upload documents (pdf/docx/txt/csv/xlsx)", accept_multiple_files=True, type=["pdf","docx","txt","csv","xls","xlsx"])
    if uploaded_files:
        if st.button("Process & Index Uploaded Files"):
            process_uploaded_files(uploaded_files)
    st.markdown("---")
    st.subheader("LLM Settings")
    st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.temperature, key="temperature_slider")
    st.session_state.selected_model = st.selectbox("Model", ["gemma2-9b-it", "mixtral-8x7b-32768", "llama3-8b-8192"], index=0, key="model_select")
    st.markdown("---")
    st.subheader("Diagnostics")
    if EMBEDDINGS_OK:
        st.success("Embeddings (HuggingFace) available")
    else:
        st.warning("Embeddings not available — using TF-IDF fallback")
    if QDRANT_OK:
        st.success("Qdrant available")
    else:
        st.info("Qdrant not available — using TF-IDF fallback store")
    st.markdown("<div class='footer'>Made with ❤️ using Streamlit & LangChain</div>", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center'>🤖 SAP Ariba RAG Chatbot</h1>", unsafe_allow_html=True)

import re

def markdown_to_html(text):
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    return text

chat_container = st.container()
with chat_container:
    for m in st.session_state.messages:
        content = m["content"] or ""
        content_html = markdown_to_html(content)
        safe_html = content_html.replace("\n", "<br>")
        if m["role"] == "user":
            st.markdown(f"<div class='chat-card user-msg'><img src='https://ui-avatars.com/api/?name=User&background=0b93f6&color=fff' class='avatar'/>{safe_html}<div style='font-size:10px;color:#eee;margin-top:6px'>{m['timestamp']}</div></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-card assistant-msg'><img src='https://ui-avatars.com/api/?name=AI&background=f1f3f4&color=222' class='avatar'/>{safe_html}<div style='font-size:10px;color:#666;margin-top:6px'>{m['timestamp']}</div></div>", unsafe_allow_html=True)

chat_option_style = """
    <style>
    .chat-options-bar {
        display: flex;
        align-items: center;
        background: #181818;
        border-radius: 8px;
        padding: 8px 12px;
        margin-bottom: 10px;
        gap: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    .chat-options-bar .option-select {
        background: #232323;
        color: #fff;
        border-radius: 6px;
        border: none;
        padding: 6px 10px;
        font-size: 15px;
        margin-right: 6px;
    }
    .chat-options-bar .attach-btn {
        background: #232323;
        color: #fff;
        border: none;
        border-radius: 6px;
        padding: 8px 10px;
        font-size: 15px;
        cursor: pointer;
        margin-right: 6px;
    }
    .chat-options-bar .send-btn {
        background: #0b93f6;
        color: #fff;
        border: none;
        border-radius: 6px;
        padding: 8px 16px;
        font-size: 15px;
        cursor: pointer;
        transition: background 0.2s;
        margin-right: 6px;
    }
    .chat-options-bar .clear-btn {
        background: #444;
        color: #fff;
        border: none;
        border-radius: 6px;
        padding: 8px 16px;
        font-size: 15px;
        cursor: pointer;
    }
    </style>
"""
st.markdown(chat_option_style, unsafe_allow_html=True)

with st.container():
    chat_cols = st.columns([1.5, 1.5, 1, 4, 0.7, 0.7])
    with chat_cols[0]:
        st.selectbox(
            "Model",
            ["gemma2-9b-it", "mixtral-8x7b-32768", "llama3-8b-8192"],
            index=["gemma2-9b-it", "mixtral-8x7b-32768", "llama3-8b-8192"].index(st.session_state.selected_model) if "selected_model" in st.session_state else 0,
            key="selected_model",
            label_visibility="collapsed",
            format_func=lambda x: f"🧠 {x}"
        )
    with chat_cols[1]:
        st.selectbox(
            "Mode",
            ["Ask", "Chat", "Search"],
            index=0,
            key="chat_mode",
            label_visibility="collapsed",
            format_func=lambda x: f"💬 {x}"
        )
    with chat_cols[2]:
        attach_clicked = st.button("📎", key="attach_btn", use_container_width=True)
        if attach_clicked:
            st.session_state.show_attach = True
    with chat_cols[3]:
        st.text_input(
            "",
            key="user_input",
            placeholder="Add context (#), extensions (@), commands (/)...",
            label_visibility="collapsed"
        )
    with chat_cols[4]:
        st.button("➤", on_click=on_send, use_container_width=True, key="send_btn")
    with chat_cols[5]:
        st.button("⨉", on_click=on_clear, use_container_width=True, key="clear_btn")

if st.session_state.get("show_attach", False):
    st.file_uploader("Attach file", type=["pdf","docx","txt","csv","xls","xlsx"], key="chat_attach", accept_multiple_files=True)
    if st.session_state.get("chat_attach"):
        process_uploaded_files(st.session_state.chat_attach)
        st.session_state.show_attach = False
