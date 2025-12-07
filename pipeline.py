from typing import Dict, Any, List, Tuple, Optional

from config import RagConfig
from data_models import DocumentChunk
from milvus_hybrid_index import MilvusHybridIndex
from rerank import Reranker, reciprocal_rank_fusion
from generator import LLMGenerator


class RAGPipeline:
    def __init__(
        self,
        index: MilvusHybridIndex,
        rag_cfg: RagConfig,
        reranker: Optional[Reranker] = None,
        generator: Optional[LLMGenerator] = None,
    ):
        self.index = index
        self.cfg = rag_cfg
        self.reranker = reranker or Reranker(self.cfg.rerank)
        self.generator = generator or LLMGenerator(self.cfg.generation)

    def _fuse(
        self,
        retr_candidates: List[Tuple[DocumentChunk, float]],
        reranked: List[Tuple[DocumentChunk, float]],
    ) -> List[Tuple[DocumentChunk, float]]:
        method = self.cfg.fusion.method

        if method == "merge_rerank":
            return reranked

        elif method == "rrf":
            list1 = retr_candidates
            list2 = reranked
            return reciprocal_rank_fusion(
                [list1, list2],
                k=self.cfg.fusion.topk_final,
            )

        elif method == "weighted":
            w_ret = self.cfg.fusion.weight_retrieval
            w_rer = 1.0 - w_ret

            ret_map = {c.chunk_id: s for c, s in retr_candidates}
            rer_map = {c.chunk_id: s for c, s in reranked}

            chunk_ids = set(ret_map.keys()) | set(rer_map.keys())
            fused: List[Tuple[DocumentChunk, float]] = []

            for cid in chunk_ids:
                if cid in ret_map:
                    any_chunk = next(c for c, _s in retr_candidates if c.chunk_id == cid)
                else:
                    any_chunk = next(c for c, _s in reranked if c.chunk_id == cid)

                s_ret = ret_map.get(cid, 0.0)
                s_rer = rer_map.get(cid, 0.0)

                fused_score = w_ret * s_ret + w_rer * s_rer
                fused.append((any_chunk, fused_score))

            fused.sort(key=lambda x: x[1], reverse=True)
            return fused

        else:
            raise ValueError(f"Unknown fusion method: {method}")

    def answer(self, question: str) -> Dict[str, Any]:
        retr_results = self.index.search(question, self.cfg.retrieval)
        retr_candidates = [(chunk, score) for chunk, score, _sd in retr_results]

        reranked = self.reranker.rerank(question, retr_candidates)

        fused = self._fuse(retr_candidates, reranked)

        if self.cfg.rerank.score_threshold > 0.0:
            fused = [
                (ch, sc) for ch, sc in fused if sc >= self.cfg.rerank.score_threshold
            ]

        fused = fused[: self.cfg.fusion.topk_final]

        contexts = [ch.text for ch, _s in fused]
        answer_text = self.generator.generate(question, contexts)

        return {
            "question": question,
            "answer": answer_text,
            "contexts": contexts,
            "retrieved_chunks": fused,
        }