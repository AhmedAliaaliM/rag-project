import json
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
import requests

sys.path.insert(0, "src")
from prompt_templates import build_prompt_v1, build_prompt_v2, build_prompt_v3
from reranker import rerank

CHROMA_DIR = Path("chroma_experiments")
COLLECTION_NAME = "v2_sentence_300"
EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "llama3.2:3b"

# Test on 5 questions only — enough to see quality differences
TEST_QUESTIONS = [
    "What methods are used for multilingual hate speech detection?",
    "How does federated learning protect data privacy?",
    "What is BERTopic and how does it compare to LDA?",
    "What deep learning model is used for demand forecasting?",
    "What is the difference between a data lake and a data warehouse?"
]

PROMPT_BUILDERS = {
    "v1_basic": build_prompt_v1,
    "v2_structured": build_prompt_v2,
    "v3_strict_reasoning": build_prompt_v3
}


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
        chunks.append({
            "text": doc,
            "source": meta["source"],
        })
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

        results[q] = {}
        for name, builder in PROMPT_BUILDERS.items():
            print(f"\n--- {name} ---")
            prompt = builder(q, reranked)
            answer = generate(prompt)
            print(answer)
            results[q][name] = answer

    # Save all answers for review
    out = Path("data/prompt_comparison.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nAll answers saved to {out}")
    print("\nReview the output above and pick the prompt style you prefer.")