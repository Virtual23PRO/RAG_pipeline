from __future__ import annotations 
from dataclasses import dataclass, field 
from typing import List, Dict, Any, Optional, Tuple

import re
import os
import math
import numpy as np
import pickle

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import faiss

import torch
from transformers import AutoTokenizer
from google import genai
from google.genai import types

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOKENIZER = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)


@dataclass
class RetrievalConfig:
    chunk_size: int = 512          
    chunk_overlap: int = 128
    mode: str = "hybrid"          
    topk_pre: int = 80         
    lambda_dense: float = 0.6      

    ef_search: int = 64             
    use_pq: bool = False            


@dataclass
class RerankConfig:
    topk_rerank: int = 40
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    score_threshold: float = 0.0    


@dataclass
class FusionConfig:
    method: str = "merge_rerank"    
    topk_final: int = 5
    weight_retrieval: float = 0.5   


@dataclass
class GenerationConfig:
    llm_name: str = "gemini-2.0-flash"
    max_context_tokens: int = 4096
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 0.95
    top_k: int = 40
    cite_or_abstain: bool = True

    


@dataclass
class RagConfig:
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)



@dataclass
class DocumentChunk:
    chunk_id: int
    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)



def simple_tokenize(text: str) -> List[str]:
    """
    for BM25 
    """
    return text.split()



def chunk_document(
    doc_id: str,
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[DocumentChunk]:
    """
    This function splits the text into chunks
    """
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
    """ loading text documents from a directory"""
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


class HybridIndex:

    def __init__(
        self,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_model = SentenceTransformer(embedding_model_name, device=self.device)

        self.chunks: List[DocumentChunk] = []

        self._bm25: Optional[BM25Okapi] = None
        self._bm25_tokenized_corpus: List[List[str]] = []

        self._faiss_index: Optional[faiss.Index] = None
        self._dim: Optional[int] = None
        self._embeddings: Optional[np.ndarray] = None


    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        for ch in chunks:
            ch.chunk_id = len(self.chunks)          
            self.chunks.append(ch)
            self._bm25_tokenized_corpus.append(simple_tokenize(ch.text))

    def build(self, ef_construction: int = 64, M: int = 32) -> None:
        if not self.chunks:
            raise ValueError("No chunks found – add chunked documents before calling build().")

        self._bm25 = BM25Okapi(self._bm25_tokenized_corpus) #bm25 representation

        texts = [ch.text for ch in self.chunks]
        emb = self.embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10
        emb_norm = emb / norms

        self._embeddings = emb_norm.astype("float32") #required by faiss
        self._dim = self._embeddings.shape[1]

        index = faiss.IndexHNSWFlat(self._dim, M) # M - the number of connections in the graph
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = 64

        index.add(self._embeddings) # create index
        self._faiss_index = index


    def _search_bm25(self, query: str, k: int) -> List[Tuple[int, float]]:
        if self._bm25 is None:
            raise ValueError("BM25 index has not been built. Call build() first.")
        q_tokens = simple_tokenize(query)
        scores = self._bm25.get_scores(q_tokens)
        topk_idx = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in topk_idx]

    def _search_vector(self, query: str, k: int, ef_search: int) -> List[Tuple[int, float]]:
        if self._faiss_index is None or self._embeddings is None:
            raise ValueError("Faiss index has not been built. Call build() first.")

        self._faiss_index.hnsw.efSearch = ef_search

        q_emb = self.embed_model.encode([query], convert_to_numpy=True)
        q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-10)
        q_emb = q_emb.astype("float32")

        D, I = self._faiss_index.search(q_emb, k) #distance and index
        idxs = I[0]
        scores = D[0]  
        return [(int(i), float(s)) for i, s in zip(idxs, scores)]

    def search(
        self,
        query: str,
        cfg: RetrievalConfig,
    ) -> List[Tuple[DocumentChunk, float, Dict[str, float]]]:
        k = cfg.topk_pre

        results: Dict[int, Dict[str, float]] = {}

        if cfg.mode in ("keyword", "hybrid"):
            bm25_res = self._search_bm25(query, k)
            for idx, score in bm25_res:
                if idx not in results:
                    results[idx] = {}
                results[idx]["bm25"] = score

        if cfg.mode in ("vector", "hybrid"):
            vec_res = self._search_vector(query, k, cfg.ef_search)
            for idx, score in vec_res:
                if idx not in results:
                    results[idx] = {}
                results[idx]["dense"] = score

        bm25_vals = [sd["bm25"] for sd in results.values() if "bm25" in sd]
        dense_vals = [sd["dense"] for sd in results.values() if "dense" in sd]

        if bm25_vals:
            bm_min, bm_max = min(bm25_vals), max(bm25_vals)
            bm_range = bm_max - bm_min if bm_max > bm_min else 1.0
        else:
            bm_min = 0.0
            bm_range = 1.0

        if dense_vals:
            de_min, de_max = min(dense_vals), max(dense_vals)
            de_range = de_max - de_min if de_max > de_min else 1.0
        else:
            de_min = 0.0
            de_range = 1.0

        merged: List[Tuple[int, float, Dict[str, float]]] = [] #chunk_id, combined_score, bm_25, vector -> scores
        for idx, sdict in results.items():
            bm25_score = sdict.get("bm25")
            dense_score = sdict.get("dense")

            if cfg.mode == "keyword":
                if bm25_score is None:
                    combined = 0.0
                else:
                    bm_norm = (bm25_score - bm_min) / bm_range
                    combined = bm_norm

            elif cfg.mode == "vector":
                if dense_score is None:
                    combined = 0.0
                else:
                    de_norm = (dense_score - de_min) / de_range
                    combined = de_norm

            else: 
                bm_norm = 0.0
                de_norm = 0.0
                if bm25_score is not None:
                    bm_norm = (bm25_score - bm_min) / bm_range
                if dense_score is not None:
                    de_norm = (dense_score - de_min) / de_range

                combined = (
                    cfg.lambda_dense * de_norm
                    + (1.0 - cfg.lambda_dense) * bm_norm
                )

            merged.append((idx, combined, sdict)) #chunk_id, combined_score, bm_25, vector -> scores

        merged.sort(key=lambda x: x[1], reverse=True) 
        merged = merged[:k]

        out: List[Tuple[DocumentChunk, float, Dict[str, float]]] = []
        for idx, score, sdict in merged:
            out.append((self.chunks[idx], score, sdict))

        return out
    
    def save(self, path: str) -> None:
        """Save index on disk"""
        os.makedirs(path, exist_ok=True)

        if self._embeddings is not None:
            np.save(os.path.join(path, "embeddings.npy"), self._embeddings)

        if self._faiss_index is not None:
            faiss.write_index(self._faiss_index, os.path.join(path, "index.faiss")) # graf HNSW

        bm25_path = os.path.join(path, "bm25.pkl")
        with open(bm25_path, "wb") as f:
            pickle.dump(
                {
                    "tokenized_corpus": self._bm25_tokenized_corpus,
                    "chunks": self.chunks,
                },
                f,
            )


    @classmethod
    def load(cls, path: str, embedding_model_name: str = EMBEDDING_MODEL_NAME, device: Optional[str] = None) -> "HybridIndex":
        index = cls(embedding_model_name=embedding_model_name, device=device)

        faiss_index_path = os.path.join(path, "index.faiss")
        index._faiss_index = faiss.read_index(faiss_index_path)

        emb_path = os.path.join(path, "embeddings.npy")
        if os.path.exists(emb_path):
            index._embeddings = np.load(emb_path).astype("float32")
            index._dim = index._embeddings.shape[1]

        bm25_path = os.path.join(path, "bm25.pkl")
        with open(bm25_path, "rb") as f:
            data = pickle.load(f)
        index._bm25_tokenized_corpus = data["tokenized_corpus"]
        index.chunks = data["chunks"]
        index._bm25 = BM25Okapi(index._bm25_tokenized_corpus)

        return index



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

        texts = [(query, c.text) for c, _ in candidates] #prepare structure for cross-encoder
        scores = self.model.predict(texts)  #reranking score

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
    score_dict: Dict[int, float] = {}
    chunk_ref: Dict[int, DocumentChunk] = {}

    for rank_list in lists: #iter retrival_rank, rerenking_rank
        for rank, (chunk, _score) in enumerate(rank_list):
            cid = chunk.chunk_id #one id, in particular two ranking
            chunk_ref[cid] = chunk
            score_dict[cid] = score_dict.get(cid, 0.0) + 1.0 / (k_rrf + rank + 1) # Reciprocal Rank Fusion algorithm

    items = [(chunk_ref[cid], s) for cid, s in score_dict.items()]
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:k]

class LLMGenerator:
    def __init__(self, cfg: GenerationConfig, device: Optional[str] = None):
        self.cfg = cfg

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing GEMINI_API_KEY environment variable. "
                "Set it before running the program."
            )

        self.client = genai.Client(api_key=api_key)
        self.model_name = cfg.llm_name

        self.gen_config = types.GenerateContentConfig(
        temperature=cfg.temperature,
        max_output_tokens=cfg.max_new_tokens,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        )

        print(f"Ładowanie LLM (Gemini): {self.model_name}")

    def _build_prompt(self, question: str, contexts: List[str]) -> str:
        ctx_text = ""
        for i, c in enumerate(contexts, start=1):
            ctx_text += f"[Fragment {i}]\n{c}\n\n"

        prompt = (
            "Otrzymasz pytanie użytkownika oraz listę fragmentów tekstu z różnych dokumentów.\n"
            "Twoje zadanie:\n"
            "1. Odpowiadaj WYŁĄCZNIE na podstawie podanych fragmentów. "
            "   Nie korzystaj z żadnej wiedzy zewnętrznej ani własnych domysłów.\n"
            "2. Jeśli fragmenty pozwalają odpowiedzieć na pytanie, odpowiedz po polsku w 1–3 zdaniach, "
            "   opierając się tylko na tych fragmentach.\n"
            "3. Jeśli fragmenty nie zawierają wystarczających informacji, aby odpowiedzieć, "
            "   napisz dokładnie jedno zdanie: \"Nie wiem na podstawie podanych dokumentów\".\n\n"
            "FRAGMENTY DOKUMENTÓW:\n"
            f"{ctx_text}\n"
            "PYTANIE UŻYTKOWNIKA:\n"
            f"{question}\n\n"
            "ODPOWIEDŹ (1–3 zdania po polsku, tylko na podstawie dokumentów):"
        )

        approx_max_chars = self.cfg.max_context_tokens * 4  
        if len(prompt) > approx_max_chars:
            prompt = prompt[-approx_max_chars:]

        return prompt

    def generate(self, question: str, contexts: List[str]) -> str:
        prompt = self._build_prompt(question, contexts)

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=self.gen_config,
            )

            text = (getattr(response, "text", "") or "").strip()
            if not text:
                return "[Model nie wygenerował odpowiedzi]"

            return text

        except Exception as e:
            print(f"[BŁĄD GEMINI] {e}")
            return "[Wystąpił błąd podczas generowania odpowiedzi]"


class RAGPipeline:
    def __init__(
        self,
        index: HybridIndex,
        rag_cfg: RagConfig,
        reranker: Optional[Reranker] = None,
        generator: Optional[LLMGenerator] = None,
    ):
        self.index = index
        self.cfg = rag_cfg
        self.reranker = reranker or Reranker(self.cfg.rerank)
        self.generator = generator or LLMGenerator(self.cfg.generation)


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
                any_chunk = None
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



def build_example_index(
    cfg: RetrievalConfig,
    embed_model_name: str = EMBEDDING_MODEL_NAME,
    data_dir: Optional[str] = None,
) -> HybridIndex:
    if data_dir is not None and os.path.isdir(data_dir):
        docs = load_corpus_from_dir(data_dir)
    else:
        docs = {
            "doc1": "Python jest wysokopoziomowym językiem programowania ogólnego przeznaczenia.",
            "doc2": "RTX 4070 to karta graficzna firmy NVIDIA oparta na architekturze Ada Lovelace.",
            "doc3": "Retrieval-Augmented Generation łączy wyszukiwanie dokumentów z dużymi modelami językowymi.",
            "doc4": "Transformers to biblioteka Pythona do pracy z modelami NLP, taka jak BERT czy GPT.",
        }

    index = HybridIndex(embedding_model_name=embed_model_name)

    all_chunks: List[DocumentChunk] = []

    for doc_id, text in docs.items():
        chs = chunk_document(
            doc_id=doc_id,
            text=text,
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
        )
        all_chunks.extend(chs)

    index.add_chunks(all_chunks)
    index.build()
    return index


def main():
    rag_cfg = RagConfig()
    data_dir = "data_docs"

    if os.path.isdir("rag_index"):
        index = HybridIndex.load("rag_index")
        print("Załadowano istniejący indeks z katalogu 'rag_index'.")
    else:
        index = build_example_index(rag_cfg.retrieval, data_dir=data_dir)
        index.save("rag_index")
        print("Zbudowano nowy indeks i zapisano w 'rag_index'.")

    rag_pipeline = RAGPipeline(index, rag_cfg)

    while True:
        q = input("\nZapytanie (puste = wyjście): ").strip()
        if not q:
            break
        out = rag_pipeline.answer(q)
        print("\n=== Odpowiedź ===")
        print(out["answer"])
        print("\n--- Użyte fragmenty ---")
        for i, ctx in enumerate(out["contexts"], start=1):
            print(f"[{i}] {ctx}")


if __name__ == "__main__":
    main()