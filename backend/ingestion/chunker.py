"""Structure-aware, tokenizer-based chunking for extracted research papers."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = INPUT_DIR / "chunks.jsonl"
TOKENIZER_NAME = "BAAI/bge-small-en-v1.5"

TARGET_TOKENS = 450
MAX_TOKENS = 550
OVERLAP_TOKENS = 60


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def stable_id(paper_id: str, text: str) -> str:
    digest = hashlib.sha1(f"{paper_id}\n{text}".encode("utf-8")).hexdigest()[:16]
    return f"{paper_id}-{digest}"


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    paper_title: str
    section: str
    subsection: str
    page_start: int
    page_end: int
    element_type: str
    token_count: int
    text: str


def classify_block(text: str) -> str:
    stripped = text.strip()
    if re.search(r"(?im)^\s*\|.*\|\s*$", stripped):
        return "table"
    if re.match(r"(?i)^\s*(figure|fig\.)\s*\d+", stripped):
        return "figure_caption"
    if re.match(r"(?i)^\s*table\s*\d+", stripped):
        return "table_caption"
    if re.search(r"\\\[|\\\]|\\begin\{|\$\$", stripped):
        return "equation"
    return "text"


def parse_markdown(markdown: str) -> list[dict[str, Any]]:
    """Convert extracted Markdown into page- and heading-aware blocks."""
    blocks: list[dict[str, Any]] = []
    page = 1
    section = "Introduction"
    subsection = ""
    paragraph_lines: list[str] = []

    def flush() -> None:
        nonlocal paragraph_lines
        text = "\n".join(paragraph_lines).strip()
        paragraph_lines = []
        if text:
            blocks.append({
                "text": text,
                "page": page,
                "section": section,
                "subsection": subsection,
                "element_type": classify_block(text),
            })

    for line in markdown.splitlines():
        page_match = re.search(r"<!--\s*PAGE\s+(\d+)\s*-->", line, re.I)
        if page_match:
            flush()
            page = int(page_match.group(1))
            continue

        heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if heading:
            flush()
            level = len(heading.group(1))
            heading_text = heading.group(2).strip()
            if level <= 2:
                section = heading_text
                subsection = ""
            else:
                subsection = heading_text
            continue

        if not line.strip():
            flush()
        else:
            paragraph_lines.append(line.rstrip())

    flush()
    return blocks


def split_oversized_block(block: dict[str, Any], tokenizer: Any) -> list[dict[str, Any]]:
    """Split long paragraphs at sentence boundaries, then use token windows."""
    text = block["text"]
    if token_count(tokenizer, text) <= MAX_TOKENS:
        return [block]

    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])", text)
    pieces: list[str] = []
    current = ""

    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if current and token_count(tokenizer, candidate) > TARGET_TOKENS:
            pieces.append(current)
            current = sentence
        else:
            current = candidate

    if current:
        pieces.append(current)

    result: list[dict[str, Any]] = []
    for piece in pieces:
        if token_count(tokenizer, piece) <= MAX_TOKENS:
            item = dict(block)
            item["text"] = piece
            result.append(item)
            continue

        ids = tokenizer.encode(piece, add_special_tokens=False)
        start = 0
        while start < len(ids):
            end = min(start + TARGET_TOKENS, len(ids))
            item = dict(block)
            item["text"] = tokenizer.decode(ids[start:end], skip_special_tokens=True).strip()
            result.append(item)
            if end == len(ids):
                break
            start = max(start + 1, end - OVERLAP_TOKENS)

    return result


def make_chunk(
    paper_id: str,
    paper_title: str,
    blocks: list[dict[str, Any]],
    tokenizer: Any,
) -> Chunk:
    text = "\n\n".join(block["text"] for block in blocks).strip()
    first = blocks[0]
    pages = [block["page"] for block in blocks]
    element_types = {block["element_type"] for block in blocks}
    element_type = next(iter(element_types)) if len(element_types) == 1 else "text"

    return Chunk(
        chunk_id=stable_id(paper_id, text),
        paper_id=paper_id,
        paper_title=paper_title,
        section=first["section"],
        subsection=first["subsection"],
        page_start=min(pages),
        page_end=max(pages),
        element_type=element_type,
        token_count=token_count(tokenizer, text),
        text=text,
    )


def chunk_blocks(
    paper_id: str,
    paper_title: str,
    blocks: list[dict[str, Any]],
    tokenizer: Any,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    buffer: list[dict[str, Any]] = []

    def flush_buffer() -> None:
        nonlocal buffer
        if buffer:
            chunks.append(make_chunk(paper_id, paper_title, buffer, tokenizer))
            buffer = []

    for original_block in blocks:
        for block in split_oversized_block(original_block, tokenizer):
            # Never combine unrelated sections or subsections.
            if buffer and (
                block["section"] != buffer[-1]["section"]
                or block["subsection"] != buffer[-1]["subsection"]
            ):
                flush_buffer()

            # Tables, captions, and equations stay isolated from prose chunks.
            if block["element_type"] != "text":
                flush_buffer()
                chunks.append(make_chunk(paper_id, paper_title, [block], tokenizer))
                continue

            candidate = buffer + [block]
            candidate_text = "\n\n".join(item["text"] for item in candidate)

            if buffer and token_count(tokenizer, candidate_text) > TARGET_TOKENS:
                previous_text = "\n\n".join(item["text"] for item in buffer)
                flush_buffer()

                # Carry a small overlap into the next chunk, within the same section.
                previous_ids = tokenizer.encode(previous_text, add_special_tokens=False)
                overlap_text = tokenizer.decode(
                    previous_ids[-OVERLAP_TOKENS:],
                    skip_special_tokens=True,
                ).strip()

                overlap_block = dict(block)
                if overlap_text:
                    overlap_block["text"] = overlap_text + "\n\n" + block["text"]

                if token_count(tokenizer, overlap_block["text"]) <= MAX_TOKENS:
                    buffer = [overlap_block]
                else:
                    buffer = [block]
            else:
                buffer.append(block)

    flush_buffer()
    return chunks


def title_from_markdown(markdown: str, fallback: str) -> str:
    for line in markdown.splitlines():
        match = re.match(r"^#\s+(.+?)\s*#*\s*$", line)
        if match:
            return match.group(1).strip()
    return fallback.replace("_", " ").replace("-", " ").title()


def main() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    markdown_files = sorted(
        path for path in INPUT_DIR.glob("*.md")
        if path.name.lower() != "sources.md"
    )

    if not markdown_files:
        raise FileNotFoundError(f"No extracted Markdown files found in {INPUT_DIR}")

    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    all_chunks: list[Chunk] = []

    for path in markdown_files:
        markdown = path.read_text(encoding="utf-8")
        paper_id = path.stem
        paper_title = title_from_markdown(markdown, paper_id)
        blocks = parse_markdown(markdown)
        chunks = chunk_blocks(paper_id, paper_title, blocks, tokenizer)
        all_chunks.extend(chunks)
        print(f"{path.name}: {len(blocks)} blocks -> {len(chunks)} chunks")

    with OUTPUT_FILE.open("w", encoding="utf-8") as output:
        for chunk in all_chunks:
            output.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    counts = [chunk.token_count for chunk in all_chunks]
    print(f"\nCreated {len(all_chunks)} chunks: {OUTPUT_FILE}")
    if counts:
        print(
            f"Token counts: min={min(counts)}, max={max(counts)}, "
            f"average={sum(counts) / len(counts):.1f}"
        )
        oversized = sum(count > MAX_TOKENS for count in counts)
        print(f"Chunks above {MAX_TOKENS} tokens: {oversized}")


if __name__ == "__main__":
    main()
