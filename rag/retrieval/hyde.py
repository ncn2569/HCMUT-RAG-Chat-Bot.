"""
rag/retrieval/hyde.py
=====================
HyDE (Hypothetical Document Embeddings) — kỹ thuật cải thiện Dense Search
cho các câu hỏi phức tạp (query_type == "COMPLEX").

Thay vì embed trực tiếp câu hỏi của người dùng (thường ngắn, mơ hồ),
ta dùng LLM để sinh ra một "tài liệu giả định" (hypothetical document) như thể
đó là câu trả lời. Sau đó embed tài liệu giả định này → vector gần hơn với
các tài liệu thật trong DB → retrieval chính xác hơn.

"""

import os
import re

from rag.core.container import container as _container


def generate_hypothetical_document(query, model_name=None):
    """
    Sinh ra một "tài liệu giả định" (hypothetical document) để cải thiện dense search.

    Args:
        query      : Câu hỏi của người dùng (đã được rewrite).
        model_name : Tên model LLM (mặc định lấy từ env MODEL_NAME).

    Returns:
        Chuỗi văn bản giả định, hoặc chính `query` nếu sinh thất bại.
    """
    if model_name is None:
        model_name = os.getenv("model_name")

    prompt = f"""Bạn đang hỗ trợ hệ thống tìm kiếm tài liệu.

    Hãy đóng vai một chuyên gia và viết một đoạn văn bản ngắn (khoảng 2-3 câu) trực tiếp trả lời hoặc cung cấp thông tin liên quan cho câu hỏi dưới đây.
    Mục tiêu là tạo ra một "tài liệu giả định" (hypothetical document) chứa các từ khóa và ngữ cảnh tự nhiên có khả năng xuất hiện trong tài liệu thật trong cơ sở dữ liệu.

    Hướng dẫn:
    - Trả lời trực tiếp, dạng văn trần thuật cung cấp thông tin (không viết lại câu hỏi).
    - Dùng từ vựng và văn phong trang trọng, học thuật liên quan đến ngữ cảnh trường đại học.
    - Chỉ trả về nội dung đoạn văn, tuyệt đối không giải thích thêm.

    Câu hỏi gốc: {query}

    Tài liệu giả định:"""

    try:
        response = _container.client.models.generate_content(model=model_name, contents=prompt)
        hypothetical = response.text.strip() if response.text else query
        hypothetical = re.sub(
            r"^(Tài liệu giả định[:：]\s*)", "", hypothetical, flags=re.IGNORECASE
        )
    except Exception as e:
        print(f"Error ở bước Hyde (Document): {e}")
        hypothetical = query

    return hypothetical
