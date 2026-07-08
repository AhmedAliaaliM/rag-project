def build_prompt_v1(query, chunks):
    """Current baseline prompt."""
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


def build_prompt_v2(query, chunks):
    """Structured prompt — forces clear format."""
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"\n[Source {i+1}: {chunk['source']}]\n{chunk['text']}\n"

    return f"""You are a research assistant analyzing academic papers.

Using ONLY the context below, answer the question in this exact format:

ANSWER: [your direct answer in 1-2 sentences]
EVIDENCE: [quote or paraphrase the specific part of the paper that supports your answer]
SOURCE: [paper filename]
CONFIDENCE: [HIGH if context directly answers / LOW if only partially relevant]

If the context does not answer the question, respond with:
ANSWER: Insufficient information
EVIDENCE: None found
SOURCE: N/A
CONFIDENCE: LOW

Context:
{context}

Question: {query}"""


def build_prompt_v3(query, chunks):
    """Strict reasoning prompt — think before answering."""
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"\n[Source {i+1}: {chunk['source']}]\n{chunk['text']}\n"

    return f"""You are a careful research assistant. Your job is to answer 
questions strictly from the provided academic paper excerpts.

Rules:
1. ONLY use information from the context below
2. NEVER guess or use outside knowledge
3. If the answer is partial, say what you know and what is missing
4. Always cite the source filename
5. Keep answers concise and factual

Context:
{context}

Question: {query}

Think step by step:
- What does the context say about this topic?
- Does it directly answer the question?
- What is the most accurate answer I can give from this context only?

Answer:"""

def build_prompt_v4_fewshot(query, chunks):
    """Few-shot prompt — shows LLM example answers before the real question."""
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"\n[Source {i+1}: {chunk['source']}]\n{chunk['text']}\n"

    return f"""You are a research assistant analyzing academic papers.
Using ONLY the context provided, answer questions in this exact format.

Here are two examples of perfect answers:

EXAMPLE 1:
Question: What is transfer learning?
ANSWER: Transfer learning reuses a model trained on one task as the starting point for a model on a different task.
EVIDENCE: The paper states that "pre-trained models can be fine-tuned on downstream tasks with minimal data."
SOURCE: example_paper.pdf
CONFIDENCE: HIGH

EXAMPLE 2:
Question: What dataset was used in the study?
ANSWER: Insufficient information
EVIDENCE: None found
SOURCE: N/A
CONFIDENCE: LOW

Now answer this question using the context below:

Context:
{context}

Question: {query}

ANSWER:
EVIDENCE:
SOURCE:
CONFIDENCE:"""