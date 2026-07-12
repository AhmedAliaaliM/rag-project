# Week 5 Findings — Evaluation Harness & Model Comparison

## Evaluation Harness Results (llama3.2:3b, local CPU)

| Metric | Value |
|---|---|
| Source hit rate | 20/20 (100%) |
| Avg quality score | 3.40/5.0 |
| Avg retrieval time | 0.07s |
| Avg rerank time | 0.84s |
| Avg generation time | 23.48s |
| Avg total latency | 24.40s |

## Latency Breakdown Insight
Generation accounts for 96.3% of total latency.
Retrieval (0.07s) and re-ranking (0.84s) together take less than 1 second.
Conclusion: optimizing retrieval further yields minimal gains.
The only meaningful lever for speed improvement is a faster LLM.

## Quality Score Breakdown
- 4/5: 11 questions (good answers, minor gaps)
- 3/5: 6 questions (partial answers, missing detail)
- 2/5: 3 questions (Q6, Q7, Q20 — topic gaps or technical detail)
- 1/5: 0 questions (no complete failures)

## Model Comparison

| Model | Hardware | Quality Score | Avg Latency |
|---|---|---|---|
| llama3.2:3b | CPU (i5-8th gen) | 3.40/5.0 | 24.40s |
| qwen2.5:7b | T4 GPU (Colab) | 4.30/5.0 | 7.34s |

## Key Findings
- Bigger model fixed all 3 weak questions (Q6, Q7, Q20 all jumped 2→4)
- 26% quality improvement with qwen2.5:7b
- 70% latency reduction on GPU vs CPU
- For production: qwen2.5:7b on GPU is clearly superior
- For development/testing: llama3.2:3b on CPU is sufficient and free

## Conclusion
The RAG pipeline quality is primarily bottlenecked by the LLM size
and hardware, not by retrieval quality (already 100%). Investing in
GPU inference yields the highest return for quality improvement.