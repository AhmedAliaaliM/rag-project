import sys
import os
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

sys.path.insert(0, "src")

from chunk_documents import chunk_strategy_v2, load_documents
from embed_and_store import embed_and_store as _embed_store
from rag_pipeline import load_retriever, retrieve, build_prompt, generate_answer
from reranker import rerank
from prompt_templates import build_prompt_v2

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
print("Retriever ready.")


class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = 10
    rerank_top_n: Optional[int] = 4


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list
    latency_seconds: float


@app.get("/health")
def health():
    return {"status": "ok", "collection": COLLECTION_NAME}


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    import time
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    t0 = time.time()

    # Retrieve
    chunks = retrieve(
        request.question, _model, _collection, top_k=request.top_k
    )

    # Rerank
    reranked = rerank(request.question, chunks, top_n=request.rerank_top_n)

    # Generate
    prompt = build_prompt_v2(request.question, reranked)
    answer = generate_answer(prompt)

    latency = round(time.time() - t0, 2)

    sources = list(set(c["source"] for c in reranked))

    return AnswerResponse(
        question=request.question,
        answer=answer,
        sources=sources,
        latency_seconds=latency
    )


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {
        "message": f"File '{file.filename}' uploaded successfully.",
        "note": "Re-index required to include in search. "
                "Restart the server after uploading new documents."
    }


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)