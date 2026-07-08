# Week 4 Findings — Re-ranking & Prompt Engineering

## Re-ranking Results

| Config | Hit Rate | Latency |
|---|---|---|
| No re-ranking | 20/20 (100%) | 25.2s |
| Re-ranking top_n=3 | 19/20 (95%) | 19.9s |
| Re-ranking top_n=4 | 20/20 (100%) | 25.2s |

Re-ranking added no accuracy gain on this dataset (already 100%).
top_n=3 was faster but lost 1 question due to Q6 being a borderline
topic (differential privacy mentioned briefly in TinyML paper).
Selected top_n=4 to maintain 100% accuracy.

## Prompt Engineering Results

| Prompt | Quality | Issues |
|---|---|---|
| v1_basic | Good prose | Verbose, inconsistent citations |
| v2_structured | Best overall | Clean format, confidence scoring |
| v3_strict_reasoning | Overthinks | Adds noise on simple questions |
| v4_fewshot | Inconsistent | Blank fields, vague citations |

Selected v2_structured — consistent format, explicit confidence
scoring useful for Week 5 evaluation harness.

## Final Config

- Chunking: v2_sentence_300 (300 chars, 30 overlap)
- Retrieval: top_k=10 candidates
- Re-ranking: cross-encoder/ms-marco-MiniLM-L-6-v2, top_n=4
- Prompt: v2_structured (ANSWER/EVIDENCE/SOURCE/CONFIDENCE)
- LLM: llama3.2:3b via Ollama (local, free)

## Overall Progress

| Version | Hit Rate | Latency | Improvement |
|---|---|---|---|
| v0.1 baseline | 20/20 | 39.7s | - |
| v0.2 chunking | 20/20 | 25.2s | 36% faster |
| v0.3 final | 20/20 | 21.4s | 46% faster |