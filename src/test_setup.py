"""
Day 1 sanity check.
Run this after installing Ollama, pulling models, and installing requirements.txt.
If both checks print successfully, your environment is ready for Week 2.
"""

def check_ollama():
    print("Checking Ollama connection...")
    try:
        from langchain_community.llms import Ollama
        llm = Ollama(model="llama3.2:3b")
        response = llm.invoke("Reply with exactly: OK")
        print(f"  Ollama responded: {response.strip()}")
        print("  ✅ Ollama check passed.\n")
    except Exception as e:
        print(f"  ❌ Ollama check failed: {e}")
        print("  → Make sure 'ollama serve' is running and you've run "
              "'ollama pull llama3.2:3b'\n")


def check_embeddings():
    print("Checking sentence-transformers (embeddings)...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        vector = model.encode("This is a test sentence.")
        print(f"  Generated embedding of length: {len(vector)}")
        print("  ✅ Embeddings check passed.\n")
    except Exception as e:
        print(f"  ❌ Embeddings check failed: {e}")
        print("  → First run downloads the model (~80MB), needs internet once.\n")


def check_chroma():
    print("Checking Chroma (vector database)...")
    try:
        import chromadb
        client = chromadb.Client()
        collection = client.create_collection("sanity_check")
        collection.add(
            documents=["This is a test document."],
            ids=["test1"],
        )
        result = collection.query(query_texts=["test"], n_results=1)
        print(f"  Chroma query returned: {result['documents']}")
        print("  ✅ Chroma check passed.\n")
    except Exception as e:
        print(f"  ❌ Chroma check failed: {e}")


if __name__ == "__main__":
    print("=" * 50)
    print("Week 1, Day 1 — Environment Sanity Check")
    print("=" * 50 + "\n")
    check_ollama()
    check_embeddings()
    check_chroma()
    print("If all three checks passed, you're ready for Day 2.")