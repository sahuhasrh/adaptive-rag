from pathlib import Path
from collections import Counter
import re
import fitz
import pymupdf4llm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPERS_DIR = PROJECT_ROOT / "data" / "papers"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_page_text(text: str) -> str:
    """Clean extraction artifacts without aggressively removing content."""

    text = text.replace("\x00", "")
    text = text.replace("\u00a0", " ")

    # Join words broken by PDF line wrapping.
    text = re.sub(r"(?<=[a-zA-Z])-\s*\n\s*(?=[a-z])", "", text)

    # Normalize excessive whitespace.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def find_repeated_lines(page_texts: list[str]) -> set[str]:
    """
    Identify lines repeated on many pages.
    Only remove lines that appear on at least 60% of pages.
    """

    if len(page_texts) < 3:
        return set()

    counts = Counter()

    for text in page_texts:
        lines = {
            line.strip()
            for line in text.splitlines()
            if line.strip()
        }
        counts.update(lines)

    threshold = max(3, int(len(page_texts) * 0.6))

    return {
        line
        for line, count in counts.items()
        if count >= threshold and len(line) < 150
    }


def remove_repeated_lines(text: str, repeated_lines: set[str]) -> str:
    """Remove likely repeated headers and footers."""

    lines = text.splitlines()

    cleaned = [
        line for line in lines
        if line.strip() not in repeated_lines
    ]

    return "\n".join(cleaned)


def extract_pdf(pdf_path: Path) -> dict:
    """Extract one PDF into page-aware Markdown."""

    print(f"\nProcessing: {pdf_path.name}")

    document = fitz.open(pdf_path)

    if document.page_count == 0:
        raise ValueError(f"PDF has no pages: {pdf_path.name}")

    # Extract individual pages to retain page boundaries.
    raw_pages = []

    for page_number in range(document.page_count):
        page_markdown = pymupdf4llm.to_markdown(
            str(pdf_path),
            pages=[page_number],
            show_progress=False,
        )

        raw_pages.append(page_markdown)

    document_metadata = document.metadata or {}

    repeated_lines = find_repeated_lines(raw_pages)

    pages = []

    for index, raw_text in enumerate(raw_pages):
        cleaned = remove_repeated_lines(raw_text, repeated_lines)
        cleaned = clean_page_text(cleaned)

        pages.append({
            "page_number": index + 1,
            "text": cleaned,
        })

    document.close()

    full_markdown = []

    for page in pages:
        full_markdown.append(
            f"\n<!-- PAGE {page['page_number']} -->\n"
        )
        full_markdown.append(page["text"])

    markdown = "\n".join(full_markdown)

    return {
        "filename": pdf_path.name,
        "title": document_metadata.get("title") or pdf_path.stem,
        "author": document_metadata.get("author", ""),
        "page_count": len(pages),
        "pages": pages,
        "markdown": markdown,
    }


def save_extraction(result: dict) -> Path:
    """Save extracted Markdown and page-aware text."""

    output_name = Path(result["filename"]).stem + ".md"
    output_path = OUTPUT_DIR / output_name

    header = (
        f"# {result['title']}\n\n"
        f"**Source PDF:** {result['filename']}\n\n"
        f"**Author:** {result['author']}\n\n"
        f"**Pages:** {result['page_count']}\n\n"
        "---\n"
    )

    output_path.write_text(
        header + result["markdown"],
        encoding="utf-8",
    )

    return output_path


def main():
    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {PAPERS_DIR}"
        )

    extraction_summary = []

    for pdf_path in pdf_files:
        result = extract_pdf(pdf_path)
        output_path = save_extraction(result)

        extracted_chars = sum(
            len(page["text"]) for page in result["pages"]
        )

        extraction_summary.append({
            "filename": pdf_path.name,
            "pages": result["page_count"],
            "characters": extracted_chars,
            "output": output_path.name,
        })

        print(f"Saved: {output_path}")
        print(f"Pages: {result['page_count']}")
        print(f"Extracted characters: {extracted_chars}")

    print("\n========== EXTRACTION SUMMARY ==========")

    for item in extraction_summary:
        print(
            f"{item['filename']}: "
            f"{item['pages']} pages, "
            f"{item['characters']} characters"
        )

    print("\nExtraction completed.")


if __name__ == "__main__":
    main()