import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROCESSED_DIR = Path("data/processed")
CHUNKS_DIR = Path("data/chunks")


def load_documents():
    docs = []
    for txt_file in sorted(PROCESSED_DIR.glob("*.txt")):
        meta_file = PROCESSED_DIR / f"{txt_file.stem}.json"
        text = txt_file.read_text(encoding="utf-8")
        meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        docs.append({"text": text, "metadata": meta})
    return docs


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    all_chunks = []
    for doc in docs:
        chunks = splitter.split_text(doc["text"])
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:  # skip garbage chunks
                continue
            all_chunks.append({
                "text": chunk,
                "metadata": {
                    "source": doc["metadata"].get("source_file", "unknown"),
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            })
    return all_chunks


def save_chunks(chunks):
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    output = CHUNKS_DIR / "chunks.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(chunks)} chunks to {output}")


def print_stats(chunks):
    lengths = [len(c["text"]) for c in chunks]
    print(f"\nChunking stats:")
    print(f"  Total chunks : {len(chunks)}")
    print(f"  Avg length   : {sum(lengths)//len(lengths)} chars")
    print(f"  Min length   : {min(lengths)} chars")
    print(f"  Max length   : {max(lengths)} chars")
    print(f"\nSample chunk (first one):")
    print("-" * 40)
    print(chunks[0]["text"])
    print("-" * 40)
    print(f"Metadata: {chunks[0]['metadata']}")


if __name__ == "__main__":
    print("Loading documents...")
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")

    print("\nChunking...")
    chunks = chunk_documents(docs)

    print_stats(chunks)
    save_chunks(chunks)