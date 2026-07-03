"""
rag/chat/semantic_cache.py
==========================
SemanticCache — Cache thông minh dựa trên ngữ nghĩa (semantic similarity).

THAY ĐỔI SO VỚI BẢN CŨ:
  - Trước: `embedder = load_embedder()` được gọi ngay khi file này được import.
    → Tạo 1 genai.Client mới, độc lập hoàn toàn với client ở pipeline.py/agent.py.
    → RAM bị chiếm thêm không cần thiết.

  - Sau: `embedder` được inject qua constructor: `SemanticCache(embedder=...)`
    → Dùng chung client từ ResourceContainer.
    → Không tạo connection dư thừa.

  - Tương tự với reranker: thay vì `from rag.retrieval.rerank import model, tokenizer`
    (load ngay khi import), giờ dùng `get_reranker()` từ rerank.py → lazy load.

LUỒNG HOẠT ĐỘNG:
  1. check(query) :
     a. Exact match → trả về ngay (O(n), nhanh nhất).
     b. Vector search → tính cosine similarity → lấy top 5.
     c. Rerank top 5 bằng cross-encoder → nếu score >= threshold → trả về.
  2. add(query, answer) : Tính embedding rồi lưu vào cache (chạy ngầm/async).
"""

import json
import os
import threading

import numpy as np
import torch
from google.genai import types


CACHE_FILE = "data/semantic_cache.jsonl"


def _cosine_similarity(vec1, vec2) -> float:
    """Tính cosine similarity giữa 2 vector numpy."""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))


class SemanticCache:
    """
    Cache thông minh: kiểm tra xem câu hỏi hiện tại có ngữ nghĩa gần giống
    với câu hỏi đã được trả lời trước đó không, nếu có thì trả về luôn câu
    trả lời cũ mà không cần chạy lại toàn bộ RAG pipeline.

    Attributes:
        embedder  : genai.Client dùng để tính embedding cho query.
        threshold : Ngưỡng score tối thiểu để coi là "câu hỏi giống nhau".
        max_size  : Số lượng entry tối đa giữ trong cache.
    """

    def __init__(self, embedder, threshold: float = 0.85, max_size: int = 20):
        """
        Args:
            embedder  : genai.Client (inject từ ResourceContainer).
            threshold : Score tối thiểu để cache hit (0.0 → 1.0).
            max_size  : Cache sẽ tự xóa entry cũ nhất khi vượt quá giới hạn này.
        """
        self._embedder = embedder
        self.threshold = threshold
        self.max_size = max_size
        self._cache: list[dict] = []  # [{query, answer, embedding}]
        self._lock = threading.Lock()
        self._load_from_disk()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, current_query: str) -> str | None:
        """
        Kiểm tra cache. Trả về câu trả lời nếu tìm thấy hit, None nếu không.

        Args:
            current_query : Câu hỏi hiện tại của người dùng.

        Returns:
            str nếu cache hit, None nếu cache miss.
        """
        with self._lock:
            cache_copy = list(self._cache)

        if not cache_copy:
            return None

        # Bước 1: Exact match (O(n), nhanh nhất — không cần gọi API)
        for item in cache_copy:
            if item["query"].lower().strip() == current_query.lower().strip():
                return item["answer"]

        # Bước 2: Vector search — tính embedding, lấy top 5 gần nhất
        try:
            res = self._embedder.models.embed_content(
                model=os.getenv("model_embedding_name"),
                contents=current_query,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
            )
            current_embedding = res.embeddings[0].values
        except Exception as e:
            print(f"[SemanticCache] Lỗi embedding khi check: {e}")
            return None

        similarities = [
            (idx, _cosine_similarity(current_embedding, item["embedding"]))
            for idx, item in enumerate(cache_copy)
            if "embedding" in item
        ]

        if not similarities:
            return None

        similarities.sort(key=lambda x: x[1], reverse=True)
        top_5_indices = [idx for idx, _ in similarities[:5]]

        # Bước 3: Rerank top 5 bằng cross-encoder để quyết định chính xác hơn
        best_score, best_original_idx = self._rerank_cache(
            current_query, cache_copy, top_5_indices
        )

        if best_score >= self.threshold:
            return cache_copy[best_original_idx]["answer"]

        return None

    def add(self, query: str, answer: str) -> None:
        """
        Thêm entry mới vào cache. Chạy bất đồng bộ (async) để không block UI.

        Args:
            query  : Câu hỏi đã được rewrite.
            answer : Câu trả lời của LLM.
        """
        # Không cache câu trả lời lỗi
        if "Có lỗi xảy ra" in answer or "Xin lỗi" in answer:
            return
        threading.Thread(
            target=self._add_async, args=(query, answer), daemon=True
        ).start()

    def flush(self) -> None:
        """Xóa sạch toàn bộ cache (cả RAM lẫn file disk)."""
        with self._lock:
            self._cache = []
            if os.path.exists(CACHE_FILE):
                os.remove(CACHE_FILE)
        print("[SemanticCache] Đã xóa sạch cache!")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _rerank_cache(
        self,
        current_query: str,
        cache_copy: list[dict],
        top_indices: list[int],
    ) -> tuple[float, int]:
        """
        Dùng cross-encoder reranker để tính score giữa current_query
        và các query trong cache.

        Returns:
            Tuple (best_score, best_original_idx_in_cache_copy).
        """
        from rag.retrieval.rerank import get_reranker

        model, tokenizer = get_reranker()
        pairs = [[current_query, cache_copy[idx]["query"]] for idx in top_indices]

        with torch.no_grad():
            inputs = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=128,
            )
            raw_scores = model(**inputs, return_dict=True).logits.view(-1).float()
            scores = torch.sigmoid(raw_scores).tolist()

        if not isinstance(scores, list):
            scores = [scores]

        best_score = max(scores)
        best_local_idx = scores.index(best_score)
        return best_score, top_indices[best_local_idx]

    def _add_async(self, query: str, answer: str) -> None:
        """Tính embedding rồi ghi vào cache (chạy trong background thread)."""
        try:
            res = self._embedder.models.embed_content(
                model=os.getenv("model_embedding_name"),
                contents=query,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
            )
            query_embedding = res.embeddings[0].values
            entry = {"query": query, "answer": answer, "embedding": query_embedding}

            with self._lock:
                already_exists = any(
                    item["query"].lower().strip() == query.lower().strip()
                    for item in self._cache
                )
                if not already_exists:
                    self._cache.append(entry)
                    if len(self._cache) > self.max_size:
                        self._cache.pop(0)
                    self._save_to_disk()
            print("[SemanticCache] Đã cập nhật cache!")
        except Exception as e:
            print(f"[SemanticCache] Lỗi khi add: {e}")

    def _load_from_disk(self) -> None:
        """Load cache từ file JSONL khi khởi động."""
        if not os.path.exists(CACHE_FILE):
            return
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self._cache.append(json.loads(line))
        # Cắt bớt nếu vượt max_size
        if len(self._cache) > self.max_size:
            self._cache = self._cache[-self.max_size:]

    def _save_to_disk(self) -> None:
        """Ghi cache xuống file JSONL (gọi trong lock)."""
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            for entry in self._cache:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def __repr__(self):
        return (
            f"SemanticCache("
            f"size={len(self._cache)}/{self.max_size}, "
            f"threshold={self.threshold})"
        )
