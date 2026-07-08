import json
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
import requests

sys.path.insert(0, "src")
from prompt_templates import build_prompt_v2, build_prompt_v4_fewshot
from reranker import rerank

CHROMA_DIR = Path("chroma_experiments")
COLLECTION_NAME = "v2_sentence_300"
EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "llama3.2:3b"

TEST_QUESTIONS = [
    "What methods are used for multilingual hate speech detection?",
    "What deep learning model is used for demand forecasting?",
    "What is the difference between a data lake and a data warehouse?",
]


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


def generate(prompt):
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
            timeout=120
        )
        return response.json()["response"]
    except Exception as e:
        return f"ERROR: {e}"


if __name__ == "__main__":
    print("Loading retriever...")
    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    results = {}

    for q in TEST_QUESTIONS:
        print(f"\n{'='*60}")
        print(f"Question: {q}")
        print('='*60)

        chunks = retrieve(q, model, collection)
        reranked = rerank(q, chunks, top_n=4)

        print("\n--- v2_structured (current best) ---")
        answer_v2 = generate(build_prompt_v2(q, reranked))
        print(answer_v2)

        print("\n--- v4_fewshot ---")
        answer_v4 = generate(build_prompt_v4_fewshot(q, reranked))
        print(answer_v4)

        results[q] = {
            "v2_structured": answer_v2,
            "v4_fewshot": answer_v4
        }

    out = Path("data/fewshot_comparison.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out}")