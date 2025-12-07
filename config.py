from dataclasses import dataclass, field


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

@dataclass
class RetrievalConfig:
    chunk_size: int = 512
    chunk_overlap: int = 128
    mode: str = "hybrid"
    topk_pre: int = 80
    lambda_dense: float = 0.6

    ef_search: int = 64

@dataclass
class RerankConfig:
    topk_rerank: int = 40
    model_name: str = RERANK_MODEL_NAME
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