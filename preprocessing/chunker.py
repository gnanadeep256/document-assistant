import json
import re
from pathlib import Path
from typing import List, Dict
from preprocessing.section_utils import extract_section_id, section_parents

# ---------------- Paths ---------------- #

PROCESSED_DIR = Path("data/processed")
PAGES_FILE = PROCESSED_DIR / "pages.json"
CHUNKS_FILE = PROCESSED_DIR / "chunks.json"

# ---------------- Chunk size ---------------- #

MIN_CHUNK_CHARS = 250
TARGET_CHUNK_CHARS = 400
MAX_CHUNK_CHARS = 700

# ---------------- Section heuristics ---------------- #

SECTION_INLINE_PATTERN = re.compile(
    r"(?<!\w)(\d+(\.\d+)*)(\s+)([A-Z][A-Za-z0-9 \-]{3,80})"
)

ALL_CAPS_PATTERN = re.compile(r"^[A-Z][A-Z\s]{5,}$")


def extract_section_from_text(text: str):
    match = SECTION_INLINE_PATTERN.search(text)
    if match:
        return f"{match.group(1)} {match.group(4)}"

    if ALL_CAPS_PATTERN.match(text.strip()):
        return text.strip()

    return None


def infer_section_level(title: str) -> int:
    if re.match(r"^\d+\.\d+\.\d+", title):
        return 3
    if re.match(r"^\d+\.\d+", title):
        return 2
    if re.match(r"^\d+\s+", title):
        return 1
    return 1


# ---------------- Text splitting ---------------- #

def split_into_blocks(text: str) -> List[str]:
    blocks = re.split(r"\n{2,}", text)

    refined = []
    for block in blocks:
        block = block.strip()
        if len(block) < 40:
            continue

        if len(block) > MAX_CHUNK_CHARS:
            sentences = re.split(r"(?<=[.])\s+(?=[A-Z])", block)
            for s in sentences:
                if len(s.strip()) >= 40:
                    refined.append(s.strip())
        else:
            refined.append(block)

    return refined


# ---------------- Chunking logic ---------------- #

def chunk_pages(pages: List[Dict]) -> List[Dict]:
    chunks = []
    chunk_id = 0

    current_section = None
    current_level = None

    buffer = ""
    buffer_pages = set()

    def flush():
        nonlocal chunk_id, buffer, buffer_pages
        if len(buffer) < MIN_CHUNK_CHARS:
            return

        section_id = extract_section_id(current_section)

        chunks.append({
            "id": f"chunk_{chunk_id:05d}",
            "pages": sorted(buffer_pages),
            "section_title": current_section or "UNKNOWN",
            "section_id": section_id,
            "section_parents": section_parents(section_id) if section_id else [],
            "section_level": current_level if current_section else -1,
            "structure_confidence": 0.9 if current_section else 0.2,
            "text": buffer.strip()
        })

        chunk_id += 1
        buffer = ""
        buffer_pages = set()

    for page in pages:
        page_num = page["page"]
        blocks = split_into_blocks(page["text"])

        for block in blocks:
            detected_section = extract_section_from_text(block)

            if detected_section:
                flush()
                current_section = detected_section
                current_level = infer_section_level(detected_section)
                continue

            buffer += " " + block
            buffer_pages.add(page_num)

            if len(buffer) >= TARGET_CHUNK_CHARS:
                flush()

    flush()
    return chunks


# ---------------- Main ---------------- #

def main():
    if not PAGES_FILE.exists():
        print("pages.json not found. Run pdf_loader first.")
        return

    with open(PAGES_FILE, "r", encoding="utf-8") as f:
        pages = json.load(f)

    chunks = chunk_pages(pages)

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    structured = sum(1 for c in chunks if c["section_title"] != "UNKNOWN")

    print(f"Created {len(chunks)} chunks")
    print(f"Structured chunks: {structured}")
    print(f"Saved to {CHUNKS_FILE}")


if __name__ == "__main__":
    main()
