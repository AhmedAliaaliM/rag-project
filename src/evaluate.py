import json
import time
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
import requests

CHROMA_DIR = Path("chroma_experiments")
COLLECTION_NAME = "v2_sentence_300"
EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "llama3.2:3b"
EVAL_FILE = Path("data/eval_set.json")
RESULTS_FILE = Path("data/eval_results.json")
TOP_K = 5


def load_retriever():
    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    return model, collection


def retrieve(query, model, collection):
    query_embedding = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K
    )
    chunks = []
    for doc, meta in zip(
        results["documents"][0],
        results["metadatas"][0]
    ):
        chunks.append({
            "text": doc,
            "source": meta["source"],
            "chunk_index": meta["chunk_index"]
        })
    return chunks


def build_prompt(query, chunks):
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"\n[Source {i+1}: {chunk['source']}]\n{chunk['text']}\n"

    return f"""You are a research assistant. Answer the question using ONLY 
the context provided below. If the context does not contain enough information 
to answer, say "I don't have enough information in the provided papers to 
answer this."

Always mention which source paper your answer comes from.

Context:
{context}

Question: {query}

Answer:"""


def generate_answer(prompt):
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
            timeout=120
        )
        return response.json()["response"]
    except Exception as e:
        return f"ERROR: {e}"


def check_source_hit(retrieved_chunks, expected_source):
    sources = [c["source"] for c in retrieved_chunks]
    return expected_source in sources


def run_eval():
    print("Loading eval set...")
    with open(EVAL_FILE, encoding="utf-8") as f:
        eval_set = json.load(f)

    print("Loading retriever...")
    model, collection = load_retriever()

    results = []
    source_hits = 0
    total = len(eval_set)

    print(f"\nRunning {total} eval questions...\n")
    print("=" * 60)

    for item in eval_set:
        qid = item["id"]
        question = item["question"]
        expected_source = item["expected_source"]

        # Retrieve
        chunks = retrieve(question, model, collection)
        source_hit = check_source_hit(chunks, expected_source)
        if source_hit:
            source_hits += 1

        # Generate
        prompt = build_prompt(question, chunks)
        t0 = time.time()
        answer = generate_answer(prompt)
        latency = round(time.time() - t0, 1)

        # Print summary
        hit_str = "HIT" if source_hit else "MISS"
        print(f"Q{qid:02d} [{hit_str}] ({latency}s) {question[:60]}...")
        if not source_hit:
            retrieved = list(set(c["source"] for c in chunks))
            print(f"     Expected: {expected_source}")
            print(f"     Got:      {retrieved}")

        results.append({
            "id": qid,
            "question": question,
            "expected_source": expected_source,
            "source_hit": source_hit,
            "retrieved_sources": list(set(c["source"] for c in chunks)),
            "answer": answer,
            "latency_seconds": latency
        })

    # Summary
    precision = source_hits / total * 100
    avg_latency = sum(r["latency_seconds"] for r in results) / total

    print("\n" + "=" * 60)
    print(f"EVAL SUMMARY")
    print(f"  Source hit rate : {source_hits}/{total} ({precision:.1f}%)")
    print(f"  Avg latency     : {avg_latency:.1f}s per question")
    print("=" * 60)

    # Save results
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total": total,
                "source_hits": source_hits,
                "source_hit_rate_pct": round(precision, 1),
                "avg_latency_seconds": round(avg_latency, 1)
            },
            "results": results
        }, f, indent=2, ensure_ascii=False)

    print(f"\nFull results saved to {RESULTS_FILE}")
    return precision


if __name__ == "__main__":
    run_eval()