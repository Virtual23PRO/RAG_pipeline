from typing import List, Tuple, Optional, Dict

import torch
from sentence_transformers import CrossEncoder

from config import RerankConfig
from data_models import DocumentChunk


class Reranker:
    def __init__(self, cfg: RerankConfig, device: Optional[str] = None):
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CrossEncoder(cfg.model_name, device=self.device)

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[DocumentChunk, float]],
    ) -> List[Tuple[DocumentChunk, float]]:
        if not candidates:
            return []

        texts = [(query, c.text) for c, _ in candidates]
        scores = self.model.predict(texts)

        reranked: List[Tuple[DocumentChunk, float]] = []
        for (chunk, _retrieval_score), s in zip(candidates, scores):
            reranked.append((chunk, float(s)))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[: self.cfg.topk_rerank]


def reciprocal_rank_fusion(
    lists: List[List[Tuple[DocumentChunk, float]]],
    k: int,
    k_rrf: int = 60,
) -> List[Tuple[DocumentChunk, float]]:
    score_dict: Dict[int, float] = {} #result for chunk
    chunk_ref: Dict[int, DocumentChunk] = {} #chunk_id -> DocumentChunk

    for rank_list in lists:
        for rank, (chunk, _score) in enumerate(rank_list):
            cid = chunk.chunk_id
            chunk_ref[cid] = chunk
            score_dict[cid] = score_dict.get(cid, 0.0) + 1.0 / (k_rrf + rank + 1)

    items = [(chunk_ref[cid], s) for cid, s in score_dict.items()]
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:k]