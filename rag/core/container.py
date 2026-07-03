"""
rag/core/container.py
=====================
ResourceContainer — Singleton class tập trung quản lý tài nguyên nặng.

SINGLETON PATTERN:
  Dùng `__new__` để đảm bảo chỉ có duy nhất 1 instance tồn tại trong toàn bộ
  vòng đời ứng dụng. Dù gọi `ResourceContainer()` bao nhiêu lần, luôn trả về
  cùng 1 object → RAM chỉ bị chiếm 1 lần.

  Ngoài ra, ta export sẵn `container = ResourceContainer()` ở cuối file.
  Các module khác chỉ cần: `from rag.core.container import container`
"""

import json
import os

import numpy as np
from dotenv import load_dotenv
from google import genai


class ResourceContainer:
    """
    Singleton class chứa tất cả tài nguyên nặng của ứng dụng.

    Attributes:
        client    : google.genai.Client — dùng chung cho mọi LLM call.
        embedder  : Alias của client, truyền vào dense_search() để rõ ý nghĩa.
        embeddings: numpy.ndarray — ma trận vector của toàn bộ chunks dữ liệu.
        bm25      : BM25Retriever — index BM25 để sparse search.
        data      : list[dict] — danh sách chunks thô {"text": "..."}.
    """

    _instance = None  # Biến class lưu instance duy nhất

    def __new__(cls):
        """Đảm bảo chỉ tạo 1 instance duy nhất (Singleton)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Load tất cả tài nguyên. Chỉ thực thi 1 lần nhờ flag _initialized."""
        if self._initialized:
            return

        # 1. Load biến môi trường
        load_dotenv("config/.env")
        _hf_home = os.getenv("HF_HOME")
        if _hf_home is not None:
            os.environ["HF_HOME"] = _hf_home

        # 2. Khởi tạo Gemini client
        self.client = genai.Client(api_key=os.getenv("API_KEY"))

        # 3. Embedder — alias của client, dùng khi gọi embed_content()
        self.embedder = self.client

        # 4. Load vector embeddings (file .npy — phần nặng nhất)
        self.embeddings = np.load("data/vectors/vectors1.npy")

        # 5. Load BM25 Retriever — import ở đây để tránh circular import
        from rag.retrieval.bm25 import BM25Retriever
        self.bm25 = BM25Retriever("data/vectors/vectors1.jsonl")

        # 6. Load raw text data
        self.data: list[dict] = []
        with open("data/vectors/vectors1.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                if stripped := line.strip():
                    self.data.append(json.loads(stripped))

        self._initialized = True
        print(
            f"[ResourceContainer] Ready: "
            f"{len(self.data)} chunks, embeddings shape={self.embeddings.shape}"
        )

    def __repr__(self):
        return (
            f"ResourceContainer("
            f"chunks={len(self.data)}, "
            f"embeddings={self.embeddings.shape})"
        )


# Module-level singleton — import trực tiếp object này ở mọi nơi:
#   from rag.core.container import container
container = ResourceContainer()
