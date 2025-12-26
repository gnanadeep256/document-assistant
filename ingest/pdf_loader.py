import fitz  # PyMuPDF
import json
from pathlib import Path
import hashlib

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def compute_document_id(pdf_path: Path) -> str:
    """
    Stable document ID based on file content.
    Same PDF → same ID.
    """
    h = hashlib.sha1()
    with open(pdf_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()[:12]


def extract_pdf_pages(pdf_path: Path):
    doc = fitz.open(pdf_path)
    pages = []

    for i, page in enumerate(doc):
        text = page.get_text("text")

        # Normalize whitespace but KEEP line breaks
        lines = [line.strip() for line in text.split("\n")]
        lines = [line for line in lines if line]
        cleaned_text = "\n".join(lines)

        if len(cleaned_text) < 100:
            continue

        pages.append({
            "page": i + 1,
            "text": cleaned_text
        })

    return pages


def main():
    pdf_files = sorted(RAW_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF found in data/raw/")
        return

    all_documents = []

    for pdf_path in pdf_files:
        document_id = compute_document_id(pdf_path)
        print(f"Extracting: {pdf_path.name} (id={document_id})")

        pages = extract_pdf_pages(pdf_path)

        if not pages:
            print("  ⚠ No valid text extracted, skipping")
            continue

        all_documents.append({
            "document_id": document_id,
            "filename": pdf_path.name,
            "num_pages": len(pages),
            "pages": pages
        })

    if not all_documents:
        raise RuntimeError("No valid PDFs were processed")

    output_path = PROCESSED_DIR / "pages.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_documents, f, indent=2, ensure_ascii=False)

    print(f"\nProcessed {len(all_documents)} document(s)")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
