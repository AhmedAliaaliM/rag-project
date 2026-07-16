import streamlit as st
import os
import time
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "src")

from sentence_transformers import SentenceTransformer
import chromadb
from reranker import rerank
from prompt_templates import build_prompt_v2

st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon="📚",
    layout="wide"
)

CHROMA_DIR = Path("chroma_experiments")
COLLECTION_NAME = "v2_sentence_300"
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
GROQ_MODEL = "llama-3.1-8b-instant"

@st.cache_resource
def load_retriever():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    return model, client, collection


def retrieve(query, model, collection, top_k=10):
    query_embedding = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    chunks = []
    for doc, meta in zip(
        results["documents"][0],
        results["metadatas"][0]
    ):
        chunks.append({"text": doc, "source": meta["source"]})
    return chunks


def generate_with_groq(prompt):
    from groq import Groq
    api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        return "ERROR: GROQ_API_KEY not set. Add it in Streamlit secrets."
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.1
    )
    return response.choices[0].message.content


def index_uploaded_file(file_path, filename, collection):
    import pdfplumber
    import re
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    pages = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append(text)

    raw_text = "\n\n".join(pages)
    raw_text = re.sub(r"https?://\S+", "", raw_text)
    raw_text = re.sub(r"[ \t]+", " ", raw_text)
    raw_text = re.sub(r"\n{3,}", "\n\n", raw_text)
    lines = [l for l in raw_text.split("\n")
             if not re.fullmatch(r"\s*\d+\s*", l)]
    cleaned = "\n".join(lines).strip()

    if len(cleaned) < 100:
        raise ValueError("Very little text extracted")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=30,
        separators=[". ", "! ", "? ", "\n\n", "\n", " ", ""]
    )
    chunks = [c for c in splitter.split_text(cleaned) if len(c.strip()) >= 50]

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embed_model.encode(chunks).tolist()

    ids = [f"upload_{filename}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "chunk_index": i}
                 for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    return len(chunks)


# UI
st.title("📚 RAG Research Assistant")
st.caption("Ask questions about research papers — answers grounded in your documents.")

with st.expander("📋 Pre-loaded papers (21 Data Science papers)"):
    st.markdown("""
    - Multilingual Hate Speech Detection
    - Federated Learning & Privacy
    - BERTopic vs LDA Topic Modeling
    - LSTM Demand Forecasting
    - Concept Drift Detection
    - Fake News Detection
    - SHAP Explainable AI
    - TinyML for Edge Devices
    - Data Lake vs Data Warehouse
    - Customer Churn Prediction
    - Transfer Learning
    - Predictive Maintenance
    - Video Summarization with LLMs
    - Anomaly Detection in IoT
    - Human Action Recognition
    - Apache Spark Scalability
    - Speech Emotion Recognition
    - Chest X-ray CNN Classification
    - And more...

    *Upload your own PDFs to add them to the search index.*
    """)

# Load retriever
with st.spinner("Loading retriever..."):
    embed_model, chroma_client, collection = load_retriever()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Retrieval candidates", 5, 20, 10)
    rerank_top_n = st.slider("Re-rank keep", 2, 6, 4)

    st.divider()
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Add a new PDF", type=["pdf"])
    if uploaded_file:
        save_path = UPLOAD_DIR / uploaded_file.name
        with open(save_path, "wb") as f:
            shutil.copyfileobj(uploaded_file, f)
        with st.spinner("Indexing document..."):
            try:
                n = index_uploaded_file(save_path, uploaded_file.name, collection)
                st.success(f"✅ Indexed {n} chunks from '{uploaded_file.name}'")
            except Exception as e:
                st.error(f"Indexing failed: {e}")

# Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📄 Sources & Details"):
                st.write("**Sources:**", ", ".join(msg["sources"]))
                st.write("**Latency:**", f"{msg['latency']}s")

if question := st.chat_input("Ask a question about your research papers..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating answer..."):
            t0 = time.time()
            chunks = retrieve(question, embed_model, collection, top_k)
            reranked = rerank(question, chunks, top_n=rerank_top_n)
            prompt = build_prompt_v2(question, reranked)
            answer = generate_with_groq(prompt)
            latency = round(time.time() - t0, 2)
            sources = list(set(c["source"] for c in reranked))

            st.markdown(answer)
            with st.expander("📄 Sources & Details"):
                st.write("**Sources:**", ", ".join(sources))
                st.write("**Latency:**", f"{latency}s")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "latency": latency
            })