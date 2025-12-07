from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class DocumentChunk:
    chunk_id: int
    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
