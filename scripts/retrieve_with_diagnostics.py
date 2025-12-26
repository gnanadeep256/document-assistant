import sys

from retrieval.retriever import Retriever
from retrieval.query_intent import detect_intent, QueryIntent
from retrieval.aggregation import aggregate_section, aggregate_global

from llm.offline_ollama import OfflineLLM
from llm.online_gemini import OnlineGeminiLLM



def aggregate_section(results, section_id):
    """
    Robust section aggregation:
    1. Force-include the root section chunk (exact match)
    2. Then include subsections (3.1, 3.2, ...)
    3. Preserve order
    """

    root_chunks = []
    sub_chunks = []
    pages = set()

    for r in results:
        sid = r.get("section_id")
        if not sid:
            continue

        if sid == section_id:
            root_chunks.append(r)
        elif sid.startswith(section_id + "."):
            sub_chunks.append(r)

    # Sort subsections numerically (3.1 < 3.2 < 3.10)
    sub_chunks.sort(key=lambda x: x["section_id"])

    ordered = root_chunks + sub_chunks

    texts = []
    for r in ordered[:8]:
        texts.append(r["text"])
        pages.update(r["pages"])

    return {
        "section": section_id,
        "pages": sorted(pages),
        "chunks_used": len(texts),
        "text": "\n\n".join(texts),
    }

def aggregate_document_summary(results):
    """
    Used for:
    - summarize this paper
    - what does this paper propose
    - main contribution

    Strategy:
    1. Prefer early pages (intro/abstract)
    2. Prefer low section numbers
    3. Then fallback to semantic relevance
    """
    collected = []
    pages = set()

    for r in results:
        if any(p in {"1", "2"} for p in r["pages"]):
            collected.append(r["text"])
            pages.update(r["pages"])
        if len(collected) >= 6:
            break

    if len(collected) < 6:
        for r in results:
            sid = r.get("section_id")
            if sid and sid.startswith(("1", "2", "3")):
                if r["text"] not in collected:
                    collected.append(r["text"])
                    pages.update(r["pages"])
            if len(collected) >= 8:
                break

    if len(collected) < 6:
        for r in results:
            if r["text"] not in collected:
                collected.append(r["text"])
                pages.update(r["pages"])
            if len(collected) >= 10:
                break

    return {
        "pages": sorted(pages),
        "text": "\n\n".join(collected),
    }



def main():
    sys.stdout.reconfigure(encoding="utf-8")

    print("\nDocument Intelligence Copilot")
    print("-" * 45)
    print("Choose mode:")
    print("1 → Retrieval only (Non-LLM)")
    print("2 → Offline LLM (LLaMA-3 via Ollama)")
    print("3 → Online LLM (Gemini)")
    mode = input("Enter choice [1/2/3]: ").strip()

    retriever = Retriever()
    llm = None

    if mode == "2":
        llm = OfflineLLM()
        print("\nRunning in OFFLINE LLaMA-3 mode")
    elif mode == "3":
        llm = OnlineGeminiLLM()
        print("\nRunning in ONLINE GEMINI mode")
    else:
        print("\nRunning in NON-LLM mode")

    while True:
        query = input("\nEnter a query (or 'exit'): ").strip()
        if query.lower() == "exit":
            break

        intent = detect_intent(query)
        print(f"\nDetected intent: {intent.name}")

        results = retriever.search(query, k=30)
        if not results:
            print("No results found.")
            continue

        # ---------- SECTION QUERIES ---------- #
        if intent == QueryIntent.SECTION:
            section_id = "".join(c for c in query if c.isdigit() or c == ".").strip(".")
            if not section_id:
                print("Could not detect section number.")
                continue

            agg = aggregate_section(results, section_id)

            print("\n=== Section Retrieval ===")
            print(f"Section    : {agg['section']}")
            print(f"Pages      : {agg['pages']}")
            print(f"Chunks used: {agg['chunks_used']}\n")

            if agg["chunks_used"] == 0:
                print("No content found for this section.")
                continue

            if llm is None:
                print(agg["text"])
            else:
                context = f"SECTION {section_id}\n\n{agg['text']}"
                print("\nQuerying LLM...\n")
                print(llm.answer(query, context, intent.name))

            print("-" * 60)
            continue

        # ---------- DOCUMENT SUMMARY ---------- #
        if intent == QueryIntent.DOCUMENT_SUMMARY:
            agg = aggregate_document_summary(results)

            if llm is None:
                print("\n=== Retrieved Context ===\n")
                print(agg["text"])
            else:
                print("\nQuerying LLM...\n")
                print(llm.answer(query, agg["text"], intent.name))

            print("\n" + "-" * 60)
            continue

        # ---------- GENERAL / FACT / WHY / COMPARISON ---------- #
        context = "\n\n".join(r["text"] for r in results[:8])

        if llm is None:
            print("\n=== Retrieved Context ===\n")
            print(context)
        else:
            print("\nQuerying LLM...\n")
            print(llm.answer(query, context, intent.name))

        print("\n" + "-" * 60)


if __name__ == "__main__":
    main()
