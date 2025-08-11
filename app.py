import os
import math
import re
import uuid
from typing import List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import fitz  # PyMuPDF
from dotenv import load_dotenv

# --- Pinecone integration ---
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_BATCH_SIZE = 64

try:
    import pinecone
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Pinecone as PineconeStore
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    EMBEDDINGS_OK = True
    PINECONE_OK = True
except Exception as e:
    EMBEDDINGS_OK = False
    PINECONE_OK = False

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from groq import Groq
from sentence_transformers import SentenceTransformer

PRIMARY = "#0b93f6"  # Updated to match SAP Ariba blue
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GEMMA_MODEL = "gemma2-9b-it"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 50
TOP_K = 8

# --- Robot SVG for SAP Ariba Chatbot ---
ASSISTANT_SVG = """
<img src="data:image/svg+xml;utf8,
<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64' width='28' height='28'>
  <rect x='8' y='16' width='48' height='32' rx='12' fill='#0b93f6'/>
  <rect x='20' y='36' width='24' height='6' rx='3' fill='#fff' opacity='0.9'/>
  <circle cx='24' cy='32' r='5' fill='#fff'/>
  <circle cx='40' cy='32' r='5' fill='#fff'/>
  <rect x='28' y='12' width='8' height='8' rx='2' fill='#0b93f6'/>
  <rect x='16' y='24' width='32' height='16' rx='8' fill='#fff' opacity='0.1'/>
</svg>"/>
"""

st.set_page_config(page_title="SAP Ariba Chatbot", page_icon="🤖", layout="wide")
st.markdown(
    f"""
    <style>
    :root {{ --primary: {PRIMARY}; }}
    .header-card {{ background-color: {PRIMARY}; padding:16px; border-radius:10px; color: white; display:flex; gap:16px; align-items:center; }}
    .chat-user {{ background: linear-gradient(90deg, rgba(11,147,246,0.12), rgba(11,147,246,0.08)); border-left: 4px solid var(--primary); padding:10px; border-radius:10px; margin:8px 0; display:flex; gap:8px; align-items:flex-start; }}
    .chat-assistant {{ background:#f3f4f6; padding:10px; border-radius:10px; margin:8px 0; display:flex; gap:8px; align-items:flex-start; }}
    .icon-left {{ width:28px; height:28px; margin-top:2px; }}
    .quiz-card {{ border:1px solid #eee; padding:14px; border-radius:10px; margin-bottom:12px; background: #fff; box-shadow: 0 2px 6px rgba(0,0,0,0.03); }}
    .question-badge {{ display:inline-block; background:var(--primary); color:white; padding:6px 10px; border-radius:12px; font-weight:700; margin-right:10px; }}
    .correct {{ color: green; font-weight:700; }}
    .summary-card {{ border-left:6px solid var(--primary); padding:12px; border-radius:8px; background:#fff; box-shadow:0 2px 6px rgba(0,0,0,0.03); margin-bottom:12px; }}
    .small-muted {{ color:#666; font-size:13px; }}
    .active-file {{ font-weight:700; color: var(--primary); }}
    </style>
    """,
    unsafe_allow_html=True,
)

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

def extract_page_text(page):
    try:
        return page.get_text()
    except Exception:
        return ""

def extract_text_from_pdf_parallel(file_bytes: bytes, max_workers: int = 6) -> str:
    text = ""
    with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
        pages = list(pdf)
        results = [None] * len(pages)
        with ThreadPoolExecutor(max_workers=min(max_workers, len(pages))) as ex:
            futures = {ex.submit(extract_page_text, p): idx for idx, p in enumerate(pages)}
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception:
                    results[idx] = ""
        text = "\n\n".join(results)
    return text

def call_groq_chat(messages: List[dict], model: str = GEMMA_MODEL, max_tokens: int = 1024, temperature: float = 0.0) -> str:
    client = Groq(api_key=GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content

def rewrite_short_query_for_retrieval(query: str) -> str:
    if len(query.strip().split()) <= 2:
        return f"What do the provided documents say about {query.strip()}?"
    return query

_crossencoder = None
def lazy_load_crossencoder():
    global _crossencoder
    if _crossencoder is None:
        try:
            from sentence_transformers import CrossEncoder
            _crossencoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception:
            _crossencoder = None
    return _crossencoder

def rerank_with_crossencoder(query: str, docs: List[Document]) -> List[Document]:
    model = lazy_load_crossencoder()
    if not model:
        return docs
    pairs = [[query, d.page_content] for d in docs]
    scores = model.predict(pairs)
    scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in scored]

# --- Pinecone vector DB session state ---
if "pinecone_db" not in st.session_state:
    if EMBEDDINGS_OK and PINECONE_OK:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        index_name = "sap-ariba-chatbot"
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(index_name, dimension=384)
        st.session_state.pinecone_db = PineconeStore(
            index_name=index_name,
            embedding_function=embeddings,
            namespace="default"
        )
        st.session_state.embeddings = embeddings
        st.session_state._pinecone_enabled = True
    else:
        st.session_state.pinecone_db = None
        st.session_state.embeddings = None
        st.session_state._pinecone_enabled = False

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()
if "all_docs" not in st.session_state:
    st.session_state.all_docs = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []
if "recent_files" not in st.session_state:
    st.session_state.recent_files = []
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}
if "active_file" not in st.session_state:
    st.session_state.active_file = None

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712027.png", width=80)  # Robot face icon
    st.markdown(f"<h2 style='color:{PRIMARY};margin:6px 0 4px 0'>SAP Ariba Chatbot</h2>", unsafe_allow_html=True)
    st.markdown("<div style='color:#444;margin-bottom:8px;'>Upload PDFs/TXT — ask questions, summarize chapters, and generate quizzes.</div>", unsafe_allow_html=True)

    uploaded_files = st.file_uploader("Upload files (PDF / TXT)", type=["pdf", "txt"], accept_multiple_files=True)

    st.markdown("---")
    if st.button("📝 Summarize Uploaded Docs"):
        st.session_state._run_summarize = True

    if st.button("🎯 Generate Quiz (5 Qs)"):
        st.session_state._run_quiz = True

    st.markdown("---")
    st.markdown("### 📜 Recent Files (click to load history)")
    if st.session_state.get("recent_files"):
        for idx, entry in enumerate(st.session_state.recent_files[:5]):
            fname = entry.get("filename", "unknown.pdf")
            ts = entry.get("timestamp", "")
            label = f"{fname}  ({ts})"
            btn_key = f"recent_file_btn_{idx}_{fname}"
            if fname == st.session_state.active_file:
                display_label = f"**{fname}**  <span class='active-file'>({ts})</span>"
                if st.button(display_label, key=btn_key, help="Click to load this file's chat history"):
                    st.session_state.active_file = fname
                    if fname not in st.session_state.chat_histories:
                        st.session_state.chat_histories[fname] = []
                    st.session_state.chat_history = list(st.session_state.chat_histories.get(fname, []))
                    st.session_state.last_sources = []
                    st.rerun()
            else:
                if st.button(label, key=btn_key):
                    st.session_state.active_file = fname
                    if fname not in st.session_state.chat_histories:
                        st.session_state.chat_histories[fname] = []
                    st.session_state.chat_history = list(st.session_state.chat_histories.get(fname, []))
                    st.session_state.last_sources = []
                    st.rerun()
    else:
        st.markdown("<div class='small-muted'>No recent files.</div>", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("➕ New Conversation"):
        st.session_state.chat_history = []
        st.session_state.last_sources = []
        st.success("Starting a new conversation — ready when you are.")

    st.markdown("---")
    st.subheader("Diagnostics")
    if EMBEDDINGS_OK:
        st.success("Embeddings (HuggingFace) available")
    else:
        st.warning("Embeddings not available")
    if PINECONE_OK:
        st.success("Pinecone available")
    else:
        st.info("Pinecone not available — using in-memory fallback")

st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="header-card">
        <div style="flex-shrink:0;">
            <span class="icon-left">{ASSISTANT_SVG}</span>
        </div>
        <div>
            <h1 style="color:white;margin:0;">SAP Ariba Chatbot</h1>
            <p style="color:#f0f0f0;margin:4px 0 0 0;">Your SAP Ariba expert assistant — upload documents, ask questions, summarize, and quiz yourself.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("")

if st.session_state.active_file:
    st.markdown(f"**Active file:** {st.session_state.active_file}", unsafe_allow_html=True)

if uploaded_files:
    added_any = False
    for file_idx, uploaded in enumerate(uploaded_files, start=1):
        if uploaded.name in st.session_state.indexed_files:
            st.info(f"File '{uploaded.name}' already indexed in this session — skipping.")
            continue

        file_bytes = uploaded.read()
        file_size_mb = len(file_bytes) / (1024 * 1024)
        status = st.info(f"Processing file {file_idx}/{len(uploaded_files)}: {uploaded.name} ({file_size_mb:.2f} MB)")

        with st.spinner("Extracting text (parallel per-page)..."):
            try:
                text = extract_text_from_pdf_parallel(file_bytes, max_workers=6)
            except Exception:
                text = ""
                with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
                    for page in pdf:
                        text += page.get_text() + "\n\n"

        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_text(text)
        total_chunks = len(chunks)
        status.info(f"Processing file {file_idx}/{len(uploaded_files)}: {uploaded.name} — {total_chunks} chunks")

        batch_size = EMBED_BATCH_SIZE
        ids = []
        metadatas = []
        documents_texts = []

        progress = st.progress(0.0)
        for start in range(0, total_chunks, batch_size):
            end = min(start + batch_size, total_chunks)
            batch_chunks = chunks[start:end]
            for i, chunk_text in enumerate(batch_chunks):
                uid = str(uuid.uuid4())
                ids.append(uid)
                documents_texts.append(chunk_text)
                metadatas.append({"source": uploaded.name, "chunk_index": start + i})
            progress.progress(min(end / max(total_chunks, 1), 1.0))
        progress.empty()

        docs_to_add = [Document(page_content=t, metadata=m) for t, m in zip(documents_texts, metadatas)]
        if st.session_state._pinecone_enabled and st.session_state.pinecone_db is not None:
            try:
                st.session_state.pinecone_db.add_documents(docs_to_add)
            except Exception:
                pass
        else:
            st.session_state.all_docs.extend(docs_to_add)

        st.session_state.indexed_files.add(uploaded.name)
        status.success(f"Indexed '{uploaded.name}' ✅")
        added_any = True

        entry = {"filename": uploaded.name, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")}
        st.session_state.recent_files = [e for e in st.session_state.recent_files if e.get("filename") != uploaded.name]
        st.session_state.recent_files.insert(0, entry)
        st.session_state.recent_files = st.session_state.recent_files[:5]
        if uploaded.name not in st.session_state.chat_histories:
            st.session_state.chat_histories[uploaded.name] = []

        st.session_state.chat_history = []
        st.session_state.last_sources = []

    if added_any:
        st.success("All set — your SAP Ariba materials are ready! Ask a question or generate a summary/quiz.")

st.markdown("---")

def retrieve_for_query(query: str, k: int = TOP_K) -> List[Document]:
    if st.session_state._pinecone_enabled and st.session_state.pinecone_db is not None:
        try:
            docs = st.session_state.pinecone_db.similarity_search(query, k=k)
        except Exception:
            docs = []
    else:
        docs = st.session_state.all_docs[:k]
    try:
        docs = rerank_with_crossencoder(query, docs)
    except Exception:
        pass
    st.session_state.last_sources = docs
    return docs

SUMMARY_BATCH_CHUNKS = 8
SUMMARY_BATCH_LIMIT = 40

def summarize_batches(docs: List[Document], batch_size: int = SUMMARY_BATCH_CHUNKS) -> List[str]:
    partials = []
    docs_to_use = docs[:SUMMARY_BATCH_LIMIT]
    if not docs_to_use:
        return partials
    num_batches = math.ceil(len(docs_to_use) / batch_size)
    prog = st.progress(0)
    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, len(docs_to_use))
        batch = docs_to_use[start:end]
        snippets = []
        chars_limit = 900
        for d in batch:
            txt = d.page_content.strip()
            if len(txt) > chars_limit:
                txt = txt[:chars_limit] + " ..."
            snippets.append(f"Source: {d.metadata.get('source','uploaded_doc')}\n{txt}")
        context_block = "\n\n".join(snippets)
        messages = [
            {"role": "system", "content": SUMMARY_SYSTEM},
            {"role": "user", "content": f"Context:\n{context_block}"}
        ]
        try:
            partial = call_groq_chat(messages, temperature=0.3, max_tokens=700)
        except Exception:
            partial = "Partial summary failed for this batch."
        partials.append(partial)
        prog.progress((i + 1) / num_batches)
    prog.empty()
    return partials

def synthesize_final_summary(partials: List[str]) -> str:
    if not partials:
        return "The provided material does not contain enough information to summarize fully."
    joined = "\n\n".join(partials[:20])
    prompt = (
        "You are SAP Ariba Expert Assistant. Combine the following partial summaries into ONE final output in Markdown format with headings 'Summary', 'Study Notes', and 'Key Definitions'.\n\n"
        "Partial summaries:\n" + joined
    )
    messages = [{"role": "system", "content": SUMMARY_SYSTEM}, {"role": "user", "content": prompt}]
    try:
        final = call_groq_chat(messages, temperature=0.3, max_tokens=800)
    except Exception:
        final = "Failed to synthesize final summary."
    return final

if st.session_state.pop("_run_summarize", False):
    docs = retrieve_for_query("summary", k=TOP_K)
    if not docs:
        st.warning("No documents uploaded yet. Please upload files to summarize.")
    else:
        st.info("Creating a structured summary for your SAP Ariba materials...")
        partials = summarize_batches(docs, batch_size=SUMMARY_BATCH_CHUNKS)
        final_summary = synthesize_final_summary(partials)
        st.markdown("<div class='summary-card'>", unsafe_allow_html=True)
        st.markdown(final_summary, unsafe_allow_html=False)
        st.markdown("</div>", unsafe_allow_html=True)
        msg_entry = {"role": "assistant", "content": final_summary, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")}
        st.session_state.chat_history.append(msg_entry)
        if st.session_state.active_file:
            st.session_state.chat_histories.setdefault(st.session_state.active_file, []).append(msg_entry)

if st.session_state.pop("_run_quiz", False):
    docs = retrieve_for_query("quiz", k=TOP_K)
    if not docs:
        st.warning("No documents uploaded yet. Please upload files to generate a quiz.")
    else:
        snippets = [d.page_content[:1200] for d in docs]
        context_block = "\n\n".join(snippets)
        messages = [{"role": "system", "content": QUIZ_SYSTEM}, {"role": "user", "content": f"Context:\n{context_block}"}]
        try:
            raw_quiz = call_groq_chat(messages, temperature=0.5, max_tokens=700)
            msg_entry = {"role": "assistant", "content": raw_quiz, "meta": {"type": "quiz"}, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")}
            st.session_state.chat_history.append(msg_entry)
            st.session_state.chat_histories.setdefault(st.session_state.active_file or "unassigned", []).append(msg_entry)
        except Exception as e:
            st.error(f"Quiz generation failed: {e}")

user_input = st.chat_input("Ask anything about your SAP Ariba documents...")
if user_input:
    original_query = user_input.strip()
    retrieval_query = rewrite_short_query_for_retrieval(original_query)

    user_msg_entry = {"role": "user", "content": original_query, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")}
    st.session_state.chat_history.append(user_msg_entry)
    target_file = st.session_state.active_file or "unassigned"
    st.session_state.chat_histories.setdefault(target_file, []).append(user_msg_entry)

    docs = retrieve_for_query(retrieval_query, k=TOP_K)
    if not docs:
        st.warning("No documents uploaded. Please upload PDFs/TXT on the left.")
    else:
        context_items = []
        for i, d in enumerate(docs, start=1):
            snippet = d.page_content.strip()
            if len(snippet) > 1200:
                snippet = snippet[:1200] + " ..."
            src = d.metadata.get("source", "uploaded_doc")
            context_items.append(f"[{i}](#source-{i}) Source: {src}\n{snippet}")

        context_block = "\n\n".join(context_items)
        messages = [
            {"role": "system", "content": QNA_SYSTEM},
            {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {original_query}"}
        ]

        try:
            assistant_text = call_groq_chat(messages, temperature=0.0, max_tokens=512)
            assistant_msg_entry = {"role": "assistant", "content": assistant_text, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")}
            st.session_state.chat_history.append(assistant_msg_entry)
            st.session_state.chat_histories.setdefault(target_file, []).append(assistant_msg_entry)
        except Exception as e:
            st.error(f"LLM error: {e}")

for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"<div class='chat-user'><div class='icon-left'>🧑‍💼</div><div><strong>You:</strong> {msg['content']}</div></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-assistant'><div class='icon-left'>{ASSISTANT_SVG}</div><div><strong>SAP Ariba Chatbot:</strong></div></div>", unsafe_allow_html=True)
        if msg.get("meta", {}).get("type") == "quiz":
            raw = msg["content"]
            qs = re.findall(r"(Q\d+\..*?)(?=(?:Q\d+\.|$))", raw, flags=re.S)
            if not qs:
                st.markdown(raw, unsafe_allow_html=False)
            else:
                for q_i, qblock in enumerate(qs, start=1):
                    header_match = re.match(r"Q\d+\.\s*(.*?)\n", qblock)
                    qtext = header_match.group(1).strip() if header_match else qblock.strip().split("\n")[0]
                    opts = re.findall(r"^([A-D])\.\s*(.*)$", qblock, flags=re.M)
                    ans_match = re.search(r"Answer:\s*([A-D])", qblock)
                    correct = ans_match.group(1).strip() if ans_match else None

                    st.markdown("<div class='quiz-card'>", unsafe_allow_html=True)
                    st.markdown(f"<span class='question-badge'>Q{q_i}</span> **{qtext}**", unsafe_allow_html=True)
                    if opts:
                        for label, text in opts:
                            if label == correct:
                                st.markdown(f"- **{label}. {text}**  <span class='correct'>✅</span>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"- {label}. {text}", unsafe_allow_html=True)
                    else:
                        st.markdown(qblock, unsafe_allow_html=False)
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(msg["content"], unsafe_allow_html=False)

if st.session_state.last_sources:
    with st.expander("View Sources used (click a citation to jump)"):
        for idx, doc in enumerate(st.session_state.last_sources, start=1):
            st.markdown(f"<a name='source-{idx}'></a>", unsafe_allow_html=True)
            src = doc.metadata.get("source", "uploaded_doc")
            snippet = doc.page_content[:1000].replace("\n", " ")
            st.markdown(f"**[{idx}] {src}**")
            st.write(snippet)

st.markdown("</div>", unsafe_allow_html=True)
