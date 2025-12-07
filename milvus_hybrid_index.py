from typing import List, Dict, Any, Optional, Tuple
import os

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from pymilvus import (
    MilvusClient,
    FieldSchema,
    DataType,
    CollectionSchema,
)

from data_models import DocumentChunk
from config import RetrievalConfig,  EMBEDDING_MODEL_NAME
from text_utils import simple_tokenize, chunk_document, load_corpus_from_dir



COLLECTION_NAME = "rag_chunks"


class MilvusHybridIndex:

    def __init__(
        self,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        device: Optional[str] = None,
        host: str = "localhost",
        port: int = 19530,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_model = SentenceTransformer(embedding_model_name, device=self.device)

        uri = f"http://{host}:{port}"
        self.client = MilvusClient(uri=uri)

        self.chunks: List[DocumentChunk] = []
        self._bm25 = None
        self._bm25_tokenized_corpus = []
        self._vector_dim = None

        self._ensure_collection()


    def _ensure_collection(self) -> None:
        if COLLECTION_NAME not in self.client.list_collections():

            dummy_vec = self.embed_model.encode(["test"], convert_to_numpy=True)
            dim = int(dummy_vec.shape[1])
            self._vector_dim = dim

            id_field = FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
            )

            doc_id_field = FieldSchema(
                name="doc_id",
                dtype=DataType.VARCHAR,
                max_length=256,
            )

            chunk_id_field = FieldSchema(
                name="chunk_id",
                dtype=DataType.INT64,
            )

            text_field = FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=4096,
            )

            emb_field = FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=dim,
            )

            schema = CollectionSchema(
                fields=[id_field, doc_id_field, chunk_id_field, text_field, emb_field],
                auto_id=True,
                enable_dynamic_field=True,
            )

            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                schema=schema,
            )

            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                index_type="HNSW",
                metric_type="L2",
                params={"M": 16, "efConstruction": 64},
            )

            self.client.create_index(
                collection_name=COLLECTION_NAME,
                index_params=index_params,
            )

        self.client.load_collection(collection_name=COLLECTION_NAME)


    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        for ch in chunks:
            ch.chunk_id = len(self.chunks)
            self.chunks.append(ch)
            self._bm25_tokenized_corpus.append(simple_tokenize(ch.text))

    def build(self) -> None:

        if not self.chunks:
            raise ValueError("No chunks found – add chunked documents before calling build().")

        self._bm25 = BM25Okapi(self._bm25_tokenized_corpus)

        texts = [ch.text for ch in self.chunks]
        emb = self.embed_model.encode(texts, convert_to_numpy=True)
        self._vector_dim = emb.shape[1]

        rows: List[Dict[str, Any]] = []
        for ch, vec in zip(self.chunks, emb):
            rows.append(
                {
                    "doc_id": ch.doc_id,
                    "chunk_id": ch.chunk_id,
                    "text": ch.text,
                    "embedding": vec.tolist(),
                }
            )

        self.client.insert(collection_name=COLLECTION_NAME, data=rows)


    def _search_bm25(self, query: str, k: int) -> List[Tuple[int, float]]:
        if self._bm25 is None:
            raise ValueError("BM25 index has not been built. Call build() first.")
        q_tokens = simple_tokenize(query)
        scores = self._bm25.get_scores(q_tokens)
        topk_idx = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in topk_idx]


    def _search_vector(self, query: str, k: int, ef_search: int) -> List[Tuple[int, float]]:
        if self._vector_dim is None:
            raise ValueError("Vector index not built – call build() first.")

        q_emb = self.embed_model.encode([query], convert_to_numpy=True)[0].tolist()

        res = self.client.search(
            collection_name=COLLECTION_NAME,
            data=[q_emb],
            limit=k,
            search_params={"metric_type": "L2"},
            output_fields=["chunk_id"],
        )

        hits = res[0] #one query
        out: List[Tuple[int, float]] = []

        for h in hits:
            cid = int(h.get("chunk_id"))
            dist = float(h.distance)
            score = -dist   
            out.append((cid, score))

        return out

    def search(
        self,
        query: str,
        cfg: RetrievalConfig,
    ) -> List[Tuple[DocumentChunk, float, Dict[str, float]]]:
        k = cfg.topk_pre
        results: Dict[int, Dict[str, float]] = {} #int - chunks index, search metodology, score

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

        merged: List[Tuple[int, float, Dict[str, float]]] = [] #chunks_id, score after fusion, dict - search metodology, score
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

            merged.append((idx, combined, sdict))

        merged.sort(key=lambda x: x[1], reverse=True)
        merged = merged[:k]

        out: List[Tuple[DocumentChunk, float, Dict[str, float]]] = [] #DocumentChunk, Fusion, dict- > search methodology, score
        for idx, score, sdict in merged:
            out.append((self.chunks[idx], score, sdict))

        return out
    

#this is helper function from demo1 RAG pipeline
def build_example_milvus_index(
    cfg: RetrievalConfig,
    embed_model_name: str = EMBEDDING_MODEL_NAME,
    data_dir: Optional[str] = None,
) -> MilvusHybridIndex:
    if data_dir is not None and os.path.isdir(data_dir):
        docs = load_corpus_from_dir(data_dir)
    else:
        docs = {
            "doc1": "Python jest wysokopoziomowym językiem programowania ogólnego przeznaczenia.",
            "doc2": "RTX 4070 to karta graficzna firmy NVIDIA oparta na architekturze Ada Lovelace.",
            "doc3": "Retrieval-Augmented Generation łączy wyszukiwanie dokumentów z dużymi modelami językowymi.",
            "doc4": "Transformers to biblioteka Pythona do pracy z modelami NLP, taka jak BERT czy GPT.",
        }

    index = MilvusHybridIndex(embedding_model_name=embed_model_name)

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
