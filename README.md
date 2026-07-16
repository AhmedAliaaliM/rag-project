# 📚 RAG Research Assistant

A production-grade Retrieval-Augmented Generation (RAG) system that answers 
questions from research papers with cited, grounded answers.

**Live demo:** [https://rag-project-94pmet4ht9gnrpgcky3nsk.streamlit.app/]

---

## What it does

Upload any PDF document and ask questions in plain English. The system finds 
the most relevant passages across all documents and generates a direct answer 
citing the exact source — instead of searching manually through hundreds of pages.

---

## Architecture
User question
↓
Embed (all-MiniLM-L6-v2)
↓
Vector search → Chroma DB (8548 chunks from 21 papers)
↓
Re-rank (cross-encoder/ms-marco-MiniLM-L-6-v2) → top 4
↓
Structured prompt (v2 template)
↓
LLM generation (Ollama locally / Groq API deployed)
↓
Answer + source citations + latency

---

## Key findings

| Experiment | Result |
|---|---|
| Chunking strategies tested | 3 (fixed-500, sentence-300, large-1000) |
| Winner | sentence-300 — 36% faster, same accuracy |
| Baseline hit rate | 20/20 (100%) |
| Baseline latency | 39.7s (CPU, llama3.2:3b) |
| After optimization | 21.4s — 46% faster |
| Answer quality (small model) | 3.40/5.0 |
| Answer quality (large model) | 4.30/5.0 |
| Generation % of total latency | 96.3% |

---

## Tech stack

| Layer | Tool |
|---|---|
| RAG framework | LangChain |
| Vector database | Chroma |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Re-ranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM (local) | llama3.2:3b via Ollama |
| LLM (deployed) | llama-3.1-8b-instant via Groq API |
| Backend | FastAPI |
| Frontend | Streamlit |
| Containerization | Docker |
| Hosting | Streamlit Community Cloud |

---

## Run locally

```bash
# Clone and setup
git clone https://github.com/AhmedAliaaliM/rag-project
cd rag-project
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Start Ollama
ollama pull llama3.2:3b
ollama serve

# Run
streamlit run streamlit_app.py
```

---

## Project structure
rag-project/
├── src/
│   ├── load_documents.py      # PDF extraction + cleaning
│   ├── chunk_documents.py     # 3 chunking strategies
│   ├── embed_and_store.py     # Embedding + Chroma storage
│   ├── rag_pipeline.py        # Core RAG pipeline
│   ├── reranker.py            # Cross-encoder re-ranking
│   ├── prompt_templates.py    # 4 prompt templates tested
│   ├── evaluate.py            # Eval harness
│   ├── eval_harness.py        # LLM-as-judge scoring
│   ├── compare_strategies.py  # Chunking comparison
│   ├── compare_prompts.py     # Prompt comparison
│   ├── api.py                 # FastAPI backend
│   └── app.py                 # Streamlit frontend
├── data/
│   ├── raw/                   # 21 source PDFs
│   ├── processed/             # Cleaned text
│   ├── chunks/                # Chunked data
│   ├── eval_set.json          # 20 eval questions
│   └── week3_findings.md      # Experiment findings
├── streamlit_app.py           # Deployed app entry point
├── Dockerfile
└── docker-compose.yml

---

## Evaluation

Built a formal evaluation harness measuring:
- **Retrieval accuracy:** did the correct source paper appear in top results?
- **Answer quality:** LLM-as-judge scoring (1-5 rubric)
- **Latency breakdown:** retrieval vs re-ranking vs generation

**Key insight:** Generation accounts for 96.3% of total latency — 
optimizing retrieval further yields minimal gains. The bottleneck is the LLM.

---

## What I'd improve next

- Semantic chunking (group by meaning, not size)
- Hybrid search (keyword + semantic)
- GPU inference for faster generation
- Document metadata filtering (search by paper/topic)
- User toggle between pre-loaded and uploaded docs

---

## Pre-loaded dataset

21 open-access research papers from arXiv covering:
Machine Learning, NLP, Computer Vision, Big Data, Predictive Analytics, AI Ethics

All papers freely available at arxiv.org