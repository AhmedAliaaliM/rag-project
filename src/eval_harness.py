import json
import time
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
import requests

sys.path.insert(0, "src")
from reranker import rerank
from prompt_templates import build_prompt_v2

CHROMA_DIR = Path("chroma_experiments")
COLLECTION_NAME = "v2_sentence_300"
EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "llama3.2:3b"
EVAL_FILE = Path("data/eval_set.json")
RESULTS_FILE = Path("data/harness_results.json")
TOP_K = 10
RERANK_TOP_N = 4


def load_retriever():
    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    return model, collection


def retrieve(query, model, collection):
    t0 = time.time()
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
        })
    retrieval_time = time.time() - t0
    return chunks, retrieval_time


def rerank_chunks(query, chunks):
    t0 = time.time()
    reranked = rerank(query, chunks, top_n=RERANK_TOP_N)
    rerank_time = time.time() - t0
    return reranked, rerank_time


def generate_answer(prompt):
    t0 = time.time()
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
            timeout=120
        )
        answer = response.json()["response"]
    except Exception as e:
        answer = f"ERROR: {e}"
    generation_time = time.time() - t0
    return answer, generation_time


def judge_answer(question, answer, expected_answer):
    """LLM-as-judge: score the answer 1-5."""
    judge_prompt = f"""You are an evaluation judge. Score the answer below 
on a scale of 1-5 based on how well it answers the question.

Scoring rubric:
5 = Perfect, complete, accurate answer with source citation
4 = Good answer, mostly correct, minor gaps
3 = Partial answer, relevant but incomplete
2 = Mostly incorrect or missing key information
1 = Wrong, irrelevant, or hallucinated

Question: {question}
Expected answer: {expected_answer}
Actual answer: {answer}

Reply with ONLY a single number (1, 2, 3, 4, or 5) and nothing else."""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": LLM_MODEL, "prompt": judge_prompt, "stream": False},
            timeout=60
        )
        score_text = response.json()["response"].strip()
        score = int(''.join(filter(str.isdigit, score_text[:3])))
        return min(max(score, 1), 5)
    except:
        return 0


def run_harness():
    print("Loading eval set...")
    with open(EVAL_FILE, encoding="utf-8") as f:
        eval_set = json.load(f)

    print("Loading retriever...")
    model, collection = load_retriever()

    results = []
    total_retrieval_time = 0
    total_rerank_time = 0
    total_generation_time = 0
    total_judge_time = 0
    source_hits = 0
    total_score = 0
    scored_count = 0

    print(f"\nRunning {len(eval_set)} questions...\n")
    print("=" * 65)

    for item in eval_set:
        qid = item["id"]
        question = item["question"]
        expected_source = item["expected_source"]
        expected_answer = item["expected_answer"]

        # Step 1: Retrieve
        chunks, ret_time = retrieve(question, model, collection)
        total_retrieval_time += ret_time

        # Step 2: Re-rank
        reranked, rer_time = rerank_chunks(question, chunks)
        total_rerank_time += rer_time

        # Step 3: Generate
        prompt = build_prompt_v2(question, reranked)
        answer, gen_time = generate_answer(prompt)
        total_generation_time += gen_time

        # Step 4: Source hit check
        sources = [c["source"] for c in reranked]
        source_hit = expected_source in sources
        if source_hit:
            source_hits += 1

        # Step 5: Judge answer quality
        t0 = time.time()
        score = judge_answer(question, answer, expected_answer)
        total_judge_time += time.time() - t0
        if score > 0:
            total_score += score
            scored_count += 1

        total_time = ret_time + rer_time + gen_time
        hit_str = "HIT" if source_hit else "MISS"
        print(f"Q{qid:02d} [{hit_str}] Score:{score}/5 "
              f"({total_time:.1f}s) {question[:45]}...")

        results.append({
            "id": qid,
            "question": question,
            "expected_source": expected_source,
            "expected_answer": expected_answer,
            "answer": answer,
            "source_hit": source_hit,
            "retrieved_sources": sources,
            "quality_score": score,
            "latency": {
                "retrieval_seconds": round(ret_time, 2),
                "rerank_seconds": round(rer_time, 2),
                "generation_seconds": round(gen_time, 2),
                "total_seconds": round(total_time, 2)
            }
        })

    # Summary
    n = len(eval_set)
    avg_ret = total_retrieval_time / n
    avg_rer = total_rerank_time / n
    avg_gen = total_generation_time / n
    avg_total = (total_retrieval_time + total_rerank_time +
                 total_generation_time) / n
    avg_score = total_score / scored_count if scored_count else 0
    hit_rate = source_hits / n * 100

    print("\n" + "=" * 65)
    print("EVALUATION HARNESS SUMMARY")
    print("=" * 65)
    print(f"  Source hit rate    : {source_hits}/{n} ({hit_rate:.1f}%)")
    print(f"  Avg quality score  : {avg_score:.2f}/5.0")
    print(f"  Avg retrieval time : {avg_ret:.2f}s")
    print(f"  Avg rerank time    : {avg_rer:.2f}s")
    print(f"  Avg generation time: {avg_gen:.2f}s")
    print(f"  Avg total latency  : {avg_total:.2f}s")
    print("=" * 65)

    # Save
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "source_hit_rate_pct": round(hit_rate, 1),
                "avg_quality_score": round(avg_score, 2),
                "avg_retrieval_seconds": round(avg_ret, 2),
                "avg_rerank_seconds": round(avg_rer, 2),
                "avg_generation_seconds": round(avg_gen, 2),
                "avg_total_seconds": round(avg_total, 2),
                "model": LLM_MODEL
            },
            "results": results
        }, f, indent=2, ensure_ascii=False)

    print(f"\nFull results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    run_harness()