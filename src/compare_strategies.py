import json
import time
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
import requests

CHROMA_DIR = Path("chroma_experiments")
EVAL_FILE = Path("data/eval_set.json")
RESULTS_FILE = Path("data/strategy_comparison.json")
EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "llama3.2:3b"
TOP_K = 5

STRATEGIES = [
    "v1_fixed_500",
    "v2_sentence_300",
    "v3_large_1000"
]


def load_eval_set():
    with open(EVAL_FILE, encoding="utf-8") as f:
        return json.load(f)


def retrieve(query, model, collection, top_k=TOP_K):
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


def evaluate_strategy(strategy_name, eval_set, model, client):
    print(f"\nEvaluating strategy: {strategy_name}")
    print("-" * 40)

    collection = client.get_collection(strategy_name)
    hits = 0
    latencies = []

    for item in eval_set:
        chunks = retrieve(item["question"], model, collection)
        sources = [c["source"] for c in chunks]
        hit = item["expected_source"] in sources
        if hit:
            hits += 1

        # Generate answer for first 5 questions only (save time)
        if item["id"] <= 5:
            t0 = time.time()
            prompt = build_prompt(item["question"], chunks)
            generate_answer(prompt)
            latencies.append(time.time() - t0)

        status = "HIT" if hit else "MISS"
        print(f"  Q{item['id']:02d} [{status}] {item['question'][:55]}...")

    hit_rate = hits / len(eval_set) * 100
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    print(f"\n  Result: {hits}/{len(eval_set)} ({hit_rate:.1f}%) | "
          f"Avg latency (5 samples): {avg_latency:.1f}s")

    return {
        "strategy": strategy_name,
        "hits": hits,
        "total": len(eval_set),
        "hit_rate_pct": round(hit_rate, 1),
        "avg_latency_seconds": round(avg_latency, 1)
    }


if __name__ == "__main__":
    print("Loading eval set and retriever...")
    eval_set = load_eval_set()
    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    all_results = []
    for strategy in STRATEGIES:
        result = evaluate_strategy(strategy, eval_set, model, client)
        all_results.append(result)

    # Print final comparison table
    print("\n" + "=" * 55)
    print("STRATEGY COMPARISON SUMMARY")
    print("=" * 55)
    print(f"{'Strategy':<22} {'Hit Rate':>10} {'Latency':>10}")
    print("-" * 55)
    for r in all_results:
        print(f"{r['strategy']:<22} "
              f"{r['hits']}/{r['total']} ({r['hit_rate_pct']}%)".rjust(10) +
              f"  {r['avg_latency_seconds']}s")

    # Save results
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")
    print("\nWinner: pick the strategy with highest hit rate.")
    print("If tied, pick the one with lower avg latency.")