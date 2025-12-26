def grounded_qa_prompt(context: str, question: str) -> str:
    return f"""
You are a technical assistant answering questions about a document.

Using ONLY the information provided in the context below:
- Synthesize information from different parts of the context
- Explain the core idea clearly
- Briefly describe how it works
- Explain why it is important

Do NOT use any external knowledge.
If the context does not contain enough information to answer fully,
state that clearly instead of guessing.

Context:
{context}

Question:
{question}

Answer (2–4 sentences):
"""
