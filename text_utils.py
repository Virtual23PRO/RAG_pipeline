from typing import List, Dict, Optional, Any
import os

from transformers import AutoTokenizer

from data_models import DocumentChunk
from config import EMBEDDING_MODEL_NAME

TOKENIZER = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)


def simple_tokenize(text: str) -> List[str]:
    return text.split()


def chunk_document(
    doc_id: str,
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[DocumentChunk]:
    
    input_ids = TOKENIZER.encode(
        text,
        add_special_tokens=False,
    )

    chunks: List[DocumentChunk] = []
    start = 0
    chunk_id = 0

    while start < len(input_ids):
        end = start + chunk_size
        token_chunk = input_ids[start:end]

        chunk_text = TOKENIZER.decode(token_chunk, skip_special_tokens=True)

        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=chunk_text,
                metadata=metadata or {},
            )
        )
        chunk_id += 1

        if end >= len(input_ids):
            break

        start = end - chunk_overlap

    return chunks


def load_corpus_from_dir(dir_path: str) -> Dict[str, str]:
    corpus: Dict[str, str] = {}
    for fname in os.listdir(dir_path):
        if not fname.lower().endswith(".txt"):
            continue
        full_path = os.path.join(dir_path, fname)
        with open(full_path, "r", encoding="utf-8") as f:
            text = f.read()
        doc_id = os.path.splitext(fname)[0]
        corpus[doc_id] = text
    return corpus

