def aggregate_section(results, section_id, max_chunks=8):
    collected = []
    pages = set()

    for r in results:
        if r.get("section_id") and r["section_id"].startswith(section_id):
            collected.append(r["text"])
            pages.update(r["pages"])

    return {
        "section": section_id,
        "pages": sorted(pages),
        "chunks_used": len(collected),
        "text": "\n\n".join(collected[:max_chunks]),
    }


def aggregate_global(results, max_chunks=12):
    texts = []
    pages = set()

    for r in results[:max_chunks]:
        texts.append(r["text"])
        pages.update(r["pages"])

    return {
        "pages": sorted(pages),
        "text": "\n\n".join(texts),
    }
