import time
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
import requests
import json

CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "rag_papers"
EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "llama3.2:3b"
TOP_K = 5


def load_retriever():
    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    return model, collection


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
            "chunk_index": meta["chunk_index"]
        })
    return chunks


def build_prompt(query, chunks):
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"\n[Source {i+1}: {chunk['source']}]\n{chunk['text']}\n"

    prompt = f"""You are a research assistant. Answer the question using ONLY 
the context provided below. If the context does not contain enough information 
to answer, say "I don't have enough information in the provided papers to 
answer this."

Always mention which source paper your answer comes from.

Context:
{context}

Question: {query}

Answer:"""
    return prompt


def generate_answer(prompt):
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.ConnectionError:
        return "ERROR: Ollama is not running. Run 'ollama serve' in a separate terminal."
    except Exception as e:
        return f"ERROR: {e}"


def ask(query):
    print(f"\nQuestion: {query}")
    print("-" * 50)

    # Retrieve
    t0 = time.time()
    model, collection = load_retriever()
    chunks = retrieve(query, model, collection)
    retrieval_time = time.time() - t0

    print(f"Retrieved {len(chunks)} chunks in {retrieval_time:.2f}s")
    print("Sources:", list(set(c["source"] for c in chunks)))

    # Generate
    print("\nGenerating answer (this may take 30-60s on CPU)...")
    t1 = time.time()
    prompt = build_prompt(query, chunks)
    answer = generate_answer(prompt)
    generation_time = time.time() - t1

    print(f"\nAnswer (generated in {generation_time:.1f}s):")
    print("=" * 50)
    print(answer)
    print("=" * 50)

    return answer


if __name__ == "__main__":
    print("RAG Pipeline — Week 2 Day 3")
    print("Loading retriever...\n")

    # Test with 3 questions from your papers
    questions = [
        "What methods are used for multilingual hate speech detection?",
        "How does federated learning protect data privacy?",
        "What is the difference between SHAP and LIME for explainable AI?"
    ]

    for q in questions:
        ask(q)
        print("\n" + "="*60 + "\n")