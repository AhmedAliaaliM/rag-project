import json
import time
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

CHUNKS_FILE = Path("data/chunks/chunks.json")
CHROMA_DIR = Path("chroma_db")

EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "rag_papers"
BATCH_SIZE = 100  # process chunks in batches to avoid RAM issues


def load_chunks():
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        return json.load(f)


def embed_and_store(chunks):
    print(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"Connecting to Chroma...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Delete existing collection if re-running
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'")
    except:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    print(f"\nEmbedding and storing {len(chunks)} chunks in batches of {BATCH_SIZE}...")
    start = time.time()

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]

        texts = [c["text"] for c in batch]
        metadatas = [c["metadata"] for c in batch]
        ids = [f"chunk_{i + j}" for j in range(len(batch))]

        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        print(f"  Stored batch {i//BATCH_SIZE + 1}/{-(-len(chunks)//BATCH_SIZE)} "
              f"({min(i+BATCH_SIZE, len(chunks))}/{len(chunks)} chunks)")

    elapsed = time.time() - start
    print(f"\nDone! {len(chunks)} chunks embedded and stored in {elapsed:.1f}s")
    print(f"Chroma DB saved to: {CHROMA_DIR}/")


def test_retrieval(query="What is hate speech detection?"):
    print(f"\nTesting retrieval with query: '{query}'")
    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    query_embedding = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    print("\nTop 3 retrieved chunks:")
    for i, (doc, meta) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0]
    )):
        print(f"\n--- Result {i+1} ---")
        print(f"Source: {meta['source']} (chunk {meta['chunk_index']})")
        print(f"Text: {doc[:200]}...")


if __name__ == "__main__":
    print("=" * 50)
    print("Week 2 Day 2 — Embed + Store in Chroma")
    print("=" * 50 + "\n")

    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}\n")

    embed_and_store(chunks)
    test_retrieval()