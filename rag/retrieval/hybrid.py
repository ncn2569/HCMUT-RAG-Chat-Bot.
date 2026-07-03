"""
rag/retrieval/hybrid.py
=======================
HybridRetriever — Class đóng gói toàn bộ luồng retrieval.

LUỒNG BÊN TRONG search():
  1. Nếu `use_hyde=True`: dùng LLM sinh hypothetical document → embed document đó
  2. Dense search (cosine similarity trên vector embeddings)
  3. BM25 sparse search
  4. RRF fusion (kết hợp dense + hyde nếu có + bm25)
  5. Rerank top 15 kết quả bằng cross-encoder
  6. Trả về top_k chunks text + confidence score
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag.retrieval.dense_search import dense_search
from rag.retrieval.rerank import rerank
from rag.retrieval.rrf_fuse import rrf_fuse

if TYPE_CHECKING:
    from rag.core.container import ResourceContainer


class HybridRetriever:
    """
    Hybrid Retriever kết hợp Dense Search + BM25 + RRF Fusion + Reranking.

    Đây là class "cộng" tất cả kỹ thuật retrieval lại:
      - Dense Search     : Tìm theo ngữ nghĩa (semantic similarity).
      - BM25             : Tìm theo từ khóa (keyword matching).
      - HyDE (tùy chọn) : Sinh tài liệu giả định để cải thiện dense search cho
                           câu hỏi phức tạp (query_type == "COMPLEX").
      - RRF Fusion       : Kết hợp điểm số từ nhiều retriever theo công thức RRF.
      - Reranking        : Dùng cross-encoder lọc tinh top results cuối cùng.
    """

    def __init__(
        self,
        container: "ResourceContainer",
        top_k_dense: int = 10,
        top_k_bm25: int = 10,
        top_k_rerank_pool: int = 15,
        top_k_final: int = 5,
    ):
        """
        Args:
            container         : ResourceContainer chứa embedder, embeddings, bm25, data.
            top_k_dense       : Số kết quả lấy từ dense search.
            top_k_bm25        : Số kết quả lấy từ BM25 search.
            top_k_rerank_pool : Số kết quả đưa vào reranker (pool trước khi lọc).
            top_k_final       : Số kết quả cuối cùng trả về sau rerank.
        """
        self._container = container
        self.top_k_dense = top_k_dense
        self.top_k_bm25 = top_k_bm25
        self.top_k_rerank_pool = top_k_rerank_pool
        self.top_k_final = top_k_final

    def search(
        self, query: str, use_hyde: bool = False
    ) -> tuple[list[str], float]:
        """
        Tìm kiếm context liên quan đến câu truy vấn.

        Args:
            query    : Câu hỏi đã được rewrite.
            use_hyde : Nếu True, dùng HyDE để cải thiện dense search.
                       Nên bật khi query_type == "COMPLEX".

        Returns:
            Tuple (contexts, confidence_score):
              - contexts         : list[str] — danh sách đoạn văn bản liên quan.
              - confidence_score : float — điểm rerank cao nhất (0.0 → 1.0).
        """
        c = self._container  # shorthand

        # --- Bước 1: Dense search (luôn chạy) ---
        dense_result = dense_search(query, c.embedder, c.embeddings, top_k=self.top_k_dense)

        # --- Bước 2: HyDE (tùy chọn, chỉ khi use_hyde=True) ---
        # Import lazy để tránh circular import và không load client khi không cần
        if use_hyde:
            hyde_result = self._run_hyde(query, c.embedder, c.embeddings)
            rrf_list = rrf_fuse(
                dense_result, hyde_result, c.bm25.search(query, top_k=self.top_k_bm25),
                k=60, weights=[1.0, 1.0, 1.0],
            )
        else:
            rrf_list = rrf_fuse(
                dense_result, c.bm25.search(query, top_k=self.top_k_bm25),
                k=60, weights=[1.0, 1.0],
            )

        # --- Bước 3: Rerank top pool ---
        rerank_pool = rrf_list[:self.top_k_rerank_pool]
        final_list = rerank(query, rerank_pool, c.data, top_k=self.top_k_final)

        # --- Bước 4: Lấy text và confidence ---
        contexts = [c.data[idx]["text"] for idx, _ in final_list]
        confidence = final_list[0][1] if final_list else 0.0

        return contexts, confidence

    def _run_hyde(self, query: str, embedder, embeddings) -> list[tuple[int, int]]:
        """
        Sinh hypothetical document bằng LLM rồi dense search trên đó.
        Private method — chỉ được gọi nội bộ từ search().

        Returns:
            Kết quả dense search trên hypothetical document.
        """
        from rag.retrieval.hyde import generate_hypothetical_document

        hyde_doc = generate_hypothetical_document(query)
        return dense_search(
            hyde_doc, embedder, embeddings,
            top_k=self.top_k_dense, is_hyde=True,
        )

    def __repr__(self):
        return (
            f"HybridRetriever("
            f"top_k_dense={self.top_k_dense}, "
            f"top_k_bm25={self.top_k_bm25}, "
            f"top_k_final={self.top_k_final})"
        )
