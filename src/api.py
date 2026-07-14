import sys
import os
import shutil
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

load_dotenv()

# Create logs directory
Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

sys.path.insert(0, "src")
from rag_pipeline import load_retriever, retrieve, generate_answer
from reranker import rerank
from prompt_templates import build_prompt_v2

# LLM config
USE_GROQ = os.getenv("USE_GROQ", "false").lower() == "true"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama3-8b-8192"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
print(f"DEBUG OLLAMA_URL: {OLLAMA_URL}")
print(f"DEBUG USE_GROQ: {USE_GROQ}")
app = FastAPI(title="RAG Research Assistant", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CHROMA_DIR = Path("chroma_experiments")
COLLECTION_NAME = "v2_sentence_300"
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Load retriever once at startup
print("Loading retriever...")
_model, _collection = load_retriever()
print(f"Retriever ready. LLM backend: {'Groq' if USE_GROQ else 'Ollama'}")


def generate_with_groq(prompt: str) -> str:
    """Generate answer using Groq API."""
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.1
    )
    return response.choices[0].message.content


def generate_with_ollama(prompt: str) -> str:
    """Generate answer using local Ollama."""
    import requests
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": "llama3.2:3b", "prompt": prompt, "stream": False},
            timeout=120
        )
        return response.json()["response"]
    except Exception as e:
        return f"ERROR: {e}"


def generate(prompt: str) -> str:
    """Route to Groq or Ollama based on config."""
    if USE_GROQ:
        return generate_with_groq(prompt)
    return generate_with_ollama(prompt)


class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = 10
    rerank_top_n: Optional[int] = 4


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list
    latency_seconds: float
    llm_backend: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "collection": COLLECTION_NAME,
        "llm_backend": "groq" if USE_GROQ else "ollama"
    }


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    logger.info(f"Question received: {request.question[:80]}")
    t0 = time.time()

    # Retrieve
    chunks = retrieve(
        request.question, _model, _collection, top_k=request.top_k
    )

    # Rerank
    reranked = rerank(request.question, chunks, top_n=request.rerank_top_n)

    # Generate
    prompt = build_prompt_v2(request.question, reranked)
    answer = generate(prompt)

    latency = round(time.time() - t0, 2)
    sources = list(set(c["source"] for c in reranked))

    logger.info(f"Answer generated in {latency}s | Sources: {sources}")

    return AnswerResponse(
        question=request.question,
        answer=answer,
        sources=sources,
        latency_seconds=latency,
        llm_backend="groq" if USE_GROQ else "ollama"
    )


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    # Save uploaded file
    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info(f"File uploaded: {file.filename}")

    try:
        # Extract text
        import pdfplumber
        import re
        from sentence_transformers import SentenceTransformer

        pages = []
        with pdfplumber.open(save_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                pages.append(text)

        raw_text = "\n\n".join(pages)

        # Clean text
        raw_text = re.sub(r"^.*viXra.*$", "", raw_text, flags=re.MULTILINE)
        raw_text = re.sub(r"https?://\S+", "", raw_text)
        raw_text = re.sub(r"[ \t]+", " ", raw_text)
        raw_text = re.sub(r"\n{3,}", "\n\n", raw_text)
        lines = [
            line for line in raw_text.split("\n")
            if not re.fullmatch(r"\s*\d+\s*", line)
        ]
        cleaned = "\n".join(lines).strip()

        if len(cleaned) < 100:
            raise ValueError("Very little text extracted — may be a scanned PDF")

        # Chunk
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=30,
            separators=[". ", "! ", "? ", "\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_text(cleaned)
        chunks = [c for c in chunks if len(c.strip()) >= 50]

        # Embed and add to existing Chroma collection
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = embed_model.encode(chunks).tolist()

        existing_count = _collection.count()
        ids = [f"upload_{file.filename}_{i}" for i in range(len(chunks))]
        metadatas = [
            {"source": file.filename, "chunk_index": i}
            for i in range(len(chunks))
        ]

        _collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        logger.info(f"Indexed {len(chunks)} chunks from {file.filename}")

        return {
            "message": f"'{file.filename}' uploaded and indexed successfully.",
            "chunks_added": len(chunks),
            "total_chunks_in_db": existing_count + len(chunks)
        }

    except Exception as e:
        logger.error(f"Indexing failed for {file.filename}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"File saved but indexing failed: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)