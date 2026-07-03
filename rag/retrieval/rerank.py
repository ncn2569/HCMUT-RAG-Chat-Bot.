"""
rag/retrieval/rerank.py
=======================
Reranker dùng mô hình BAAI/bge-reranker-v2-m3.
"""

import torch
from functools import lru_cache
from transformers import AutoModelForSequenceClassification, AutoTokenizer

RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"


@lru_cache(maxsize=1)
def get_reranker():
    """
    Load và cache model + tokenizer của reranker.
    lru_cache(maxsize=1) đảm bảo chỉ load 1 lần duy nhất trong suốt vòng đời app.
    """
    print(f"[Reranker] Đang load model '{RERANKER_MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_NAME)
    model.eval()
    print("[Reranker] Load xong!")
    return model, tokenizer


def rerank(query, rrf_list, data, top_k=5):
    """
    Rerank danh sách kết quả từ RRF bằng cross-encoder BAAI/bge-reranker-v2-m3.

    Args:
        query    : Câu truy vấn gốc.
        rrf_list : List[(idx, rank)] từ rrf_fuse().
        data     : Danh sách chunks dữ liệu thô (list of dict {"text": ...}).
        top_k    : Số kết quả trả về sau rerank.

    Returns:
        List[(idx, score)] đã sort giảm dần theo score.
    """
    model, tokenizer = get_reranker()

    pairs = [[query, data[idx]["text"]] for idx, _ in rrf_list]

    with torch.no_grad():
        inputs = tokenizer(
            pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
        )
        scores = (
            model(**inputs, return_dict=True)
            .logits.view(-1)
            .float()
        )
        scores = torch.sigmoid(scores).tolist()

    results = []
    for i, (idx, _) in enumerate(rrf_list):
        score_val = scores[i] if isinstance(scores, list) else scores
        results.append((idx, score_val))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]
