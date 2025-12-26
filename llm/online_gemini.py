import os
from dotenv import load_dotenv
from google import genai

load_dotenv()


class OnlineGeminiLLM:
    def __init__(self, model: str = "gemini-2.5-flash"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not found in environment")

        # Correct NEW SDK usage
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def answer(self, question: str, context: str, intent: str) -> str:
        prompt = self._prompt(question, context, intent)

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            return response.text.strip()

        except Exception as e:
            return f"[GEMINI ERROR] {e}"

    def _prompt(self, question: str, context: str, intent: str) -> str:
        if intent == "DOCUMENT_SUMMARY":
            instruction = (
                "You are given multiple excerpts from a single research paper. "
                "Produce a coherent high-level summary of the paper by synthesizing "
                "the information across all excerpts."
            )

            rules = (
                "- Base your summary ONLY on the provided text\n"
                "- It is OK to combine information across multiple sections\n"
                "- Do NOT mention missing sections like abstract or introduction\n"
                "- Do NOT refuse unless the text is completely unrelated\n"
            )

        elif intent == "SECTION":
            instruction = (
                "The following text contains content from a specific numbered section of a document. "
                "Explain what this section discusses in a clear and structured way."
            )
            rules = (
                "- Treat the provided text as authoritative section content\n"
                "- Do NOT check whether the section exists\n"
                "- Do NOT refuse or say information is missing\n"
                "- Use only the given text\n"
            )


        elif intent == "COMPARISON":
            instruction = (
                "Compare the concepts mentioned using only the provided text."
            )
            rules = "- Use only the given context\n"

        elif intent == "WHY":
            instruction = "Explain the reason using only the provided text."
            rules = "- Use only the given context\n"

        else:
            instruction = "Answer the question using only the provided text."
            rules = (
                "- If the answer is missing, say:\n"
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
