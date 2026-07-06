import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

PROCESSED_DIR = Path("data/processed")
CHUNKS_DIR = Path("data/chunks")
CHROMA_BASE_DIR = Path("chroma_experiments")
EMBED_MODEL = "all-MiniLM-L6-v2"


def load_documents():
    docs = []
    for txt_file in sorted(PROCESSED_DIR.glob("*.txt")):
        meta_file = PROCESSED_DIR / f"{txt_file.stem}.json"
        text = txt_file.read_text(encoding="utf-8")
        meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        docs.append({"text": text, "metadata": meta})
    return docs


def chunk_strategy_v1(docs):
    """Fixed size: 500 chars, 50 overlap — baseline"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return _split(docs, splitter, "v1_fixed_500")


def chunk_strategy_v2(docs):
    """Sentence-aware: 300 chars, 30 overlap — keeps sentences whole"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30,
        separators=[". ", "! ", "? ", "\n\n", "\n", " ", ""]
    )
    return _split(docs, splitter, "v2_sentence_300")


def chunk_strategy_v3(docs):
    """Larger chunks: 1000 chars, 100 overlap — more context per chunk"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return _split(docs, splitter, "v3_large_1000")


def _split(docs, splitter, strategy_name):
    all_chunks = []
    for doc in docs:
        chunks = splitter.split_text(doc["text"])
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:
                continue
            all_chunks.append({
                "text": chunk,
                "metadata": {
                    "source": doc["metadata"].get("source_file", "unknown"),
                    "chunk_index": i,
                    "strategy": strategy_name
                }
            })
    return all_chunks


def embed_and_store(chunks, collection_name):
    print(f"  Embedding {len(chunks)} chunks -> collection: {collection_name}")
    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_BASE_DIR))

    try:
        client.delete_collection(collection_name)
    except:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
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
        print(f"    Batch {i//batch_size + 1}/{-(-len(chunks)//batch_size)} done")

    print(f"  Stored {len(chunks)} chunks in '{collection_name}'\n")


def save_chunks(chunks, filename):
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    out = CHUNKS_DIR / filename
    with open(out, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    print("Loading documents...")
    docs = load_documents()
    print(f"Loaded {len(docs)} documents\n")

    strategies = [
        ("v1_fixed_500",    chunk_strategy_v1),
        ("v2_sentence_300", chunk_strategy_v2),
        ("v3_large_1000",   chunk_strategy_v3),
    ]

    for name, fn in strategies:
        print(f"Running strategy: {name}")
        chunks = fn(docs)
        lengths = [len(c["text"]) for c in chunks]
        print(f"  Total chunks : {len(chunks)}")
        print(f"  Avg length   : {sum(lengths)//len(lengths)} chars")
        save_chunks(chunks, f"chunks_{name}.json")
        embed_and_store(chunks, name)

    print("All strategies done. Ready for evaluation.")