import os
from typing import List
from PyPDF2 import PdfReader
from transformers import AutoTokenizer


def extract_chunks(
    pdf_path: str,
    chunk_size: int = 100,
    tokenizer_name: str = "dbmdz/gpt2-turkish",
    chunk_overlap: int = 0,
) -> List[str]:
    """Extract text from PDF and split into token-based chunks."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(pdf_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    chunks = []
    for page in reader.pages:
        text = page.extract_text() or ""
        tokens = tokenizer.encode(text, add_special_tokens=False)
        step = max(chunk_size - chunk_overlap, 1)
        for i in range(0, len(tokens), step):
            piece = tokens[i : i + chunk_size]
            chunk = tokenizer.decode(piece)
            if chunk.strip():
                chunks.append(chunk.strip())
    return chunks
