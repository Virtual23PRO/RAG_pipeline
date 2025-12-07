from milvus_hybrid_index import build_example_milvus_index
from pipeline import RAGPipeline
from config import RagConfig


def main():
    rag_cfg = RagConfig()
    data_dir = "data_docs"

    index = build_example_milvus_index(
        rag_cfg.retrieval,
        data_dir=data_dir,
    )

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