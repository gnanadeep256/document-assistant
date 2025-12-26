import subprocess


class OfflineLLM:
    def __init__(self, model: str = "llama3.2"):
        self.model = model

    def answer(self, question: str, context: str, intent: str) -> str:
        if not context.strip():
            return "No relevant information found in the document."

        prompt = self._prompt(question, context, intent)

        try:
            result = subprocess.run(
                ["ollama", "run", self.model],
                input=prompt,
                text=True,
                encoding="utf-8",
                capture_output=True,
                timeout=120
            )
        except Exception as e:
            return f"[LLM ERROR] {e}"

        if result.returncode != 0:
            return f"[LLM ERROR] {result.stderr}"

        return result.stdout.strip()

    def _prompt(self, question: str, context: str, intent: str) -> str:
        if intent == "DOCUMENT_SUMMARY":
            instruction = (
                "You are given multiple excerpts from a single research paper. "
                "Write a concise but complete summary of the paper by synthesizing "
                "information across all excerpts."
            )

            rules = (
                "- Use ONLY the provided text\n"
                "- Combine information across sections\n"
                "- Do NOT say information is missing unless context is irrelevant\n"
            )

        elif intent == "SECTION":
            instruction = (
                "Explain and summarize the given section using only the provided text."
            )
            rules = "- Use only the given section\n"

        elif intent == "COMPARISON":
            instruction = "Compare the concepts using only the provided text."
            rules = "- Use only the given context\n"

        elif intent == "WHY":
            instruction = "Explain the reason using only the provided text."
            rules = "- Use only the given context\n"

        else:
            instruction = "Answer the question using only the provided text."
            rules = (
                "- If missing, say:\n"
                "  'The document does not contain enough information to answer this.'"
            )

        return f"""
{instruction}

Rules:
{rules}

--------------------
CONTEXT:
{context}
--------------------

QUESTION:
{question}

ANSWER:
""".strip()

