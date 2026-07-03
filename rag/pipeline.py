"""
rag/pipeline.py
===============
RAGPipeline — Class điều phối toàn bộ luồng RAG (Retrieval-Augmented Generation).

LUỒNG BÊN TRONG query():
  1. Kiểm tra Semantic Cache → trả về ngay nếu hit.
  2. Lấy lịch sử hội thoại.
  3. Rewrite + Classify query (1 API call).
  4. HybridRetriever.search() với use_hyde dựa theo query_type.
  5. Build prompt → gọi LLM → lấy câu trả lời.
  6. Lưu vào history và cache.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from rag.chat.history import add_turn, clear_history, get_history
from rag.chat.semantic_cache import SemanticCache
from rag.generation.build_prompt import build_prompt, rewrite_and_classify_query
from rag.retrieval.hybrid import HybridRetriever

if TYPE_CHECKING:
    from rag.core.container import ResourceContainer


class RAGPipeline:
    """
    Điều phối toàn bộ luồng RAG:
      Semantic Cache → Query Rewriting → HybridRetrieval → LLM Generation

    Usage:
        from rag.core.container import container
        from rag.pipeline import RAGPipeline

        pipeline = RAGPipeline(container)
        answer = pipeline.query("Điểm chuẩn ngành CNTT năm 2024 là bao nhiêu?")
    """

    def __init__(self, container: "ResourceContainer"):
        """
        Args:
            container : ResourceContainer cung cấp client, embedder, embeddings, bm25, data.
        """
        self._container = container
        self._retriever = HybridRetriever(container)
        self._cache = SemanticCache(embedder=container.embedder)
        print("[RAGPipeline] Ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, user_query: str) -> str:
        """
        API chính: nhận câu hỏi, trả về câu trả lời.
        Có đầy đủ: Semantic Cache + History + Rewrite + HyDE + Rerank.

        Args:
            user_query : Câu hỏi thô từ người dùng.

        Returns:
            Câu trả lời dạng string.
        """
        # 1. Kiểm tra cache trước — trả về ngay nếu tìm thấy
        if cached := self._cache.check(user_query):
            add_turn(user_query, cached, user_query)
            return cached

        # 2. Lấy lịch sử & rewrite + classify trong 1 API call
        history = get_history()
        rewritten, query_type = rewrite_and_classify_query(user_query, history)

        # 3. Retrieval: dùng HyDE khi câu hỏi phức tạp
        use_hyde = (query_type == "COMPLEX")
        contexts, _ = self._retriever.search(rewritten, use_hyde=use_hyde)

        # 4. Generate câu trả lời
        answer = self._generate(rewritten, contexts)

        # 5. Lưu vào history và cache
        add_turn(user_query, answer, rewritten)
        if "Có lỗi xảy ra" not in answer:
            self._cache.add(rewritten, answer)

        return answer

    def query_with_contexts(
        self, user_query: str
    ) -> tuple[str, list[str]]:
        """
        Tương tự query() nhưng trả thêm danh sách contexts đã retrieve.
        Dùng cho evaluation (RAGAS, v.v.) hoặc debugging.

        Args:
            user_query : Câu hỏi thô từ người dùng.

        Returns:
            Tuple (answer: str, contexts: list[str]).
        """
        history = get_history()
        rewritten, query_type = rewrite_and_classify_query(user_query, history)

        use_hyde = (query_type == "COMPLEX")
        contexts, _ = self._retriever.search(rewritten, use_hyde=use_hyde)

        answer = self._generate(rewritten, contexts)
        add_turn(user_query, answer, rewritten)

        return answer, contexts

    def reset_history(self) -> None:
        """Xóa sạch lịch sử hội thoại."""
        clear_history()

    # ------------------------------------------------------------------
    # Properties (read-only access cho các component bên ngoài nếu cần)
    # ------------------------------------------------------------------

    @property
    def retriever(self) -> HybridRetriever:
        """Truy cập HybridRetriever — Agent dùng để gọi search trực tiếp."""
        return self._retriever

    @property
    def cache(self) -> SemanticCache:
        """Truy cập SemanticCache — UI dùng để flush cache."""
        return self._cache

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate(self, query: str, contexts: list[str]) -> str:
        """
        Gọi LLM để sinh câu trả lời từ query và contexts.

        Args:
            query    : Câu hỏi đã rewrite.
            contexts : Danh sách đoạn văn context từ retrieval.

        Returns:
            Câu trả lời string, hoặc thông báo lỗi nếu gọi API thất bại.
        """
        prompt = build_prompt(query, contexts)
        try:
            response = self._container.client.models.generate_content(
                model=os.getenv("model_name"), contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            print(f"[RAGPipeline] Lỗi khi generate: {e}")
            return (
                "Co loi xay ra r ban oi, :)))) thu lai giup minh sau nha, "
                "co the la google dang nghen server ay ma hem sao dau. Xi nua hoi lai nhen."
            )

    def __repr__(self):
        return (
            f"RAGPipeline("
            f"retriever={self._retriever}, "
            f"cache={self._cache})"
        )
