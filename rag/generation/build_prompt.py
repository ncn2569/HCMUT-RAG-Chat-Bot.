"""
rag/generation/build_prompt.py
===============================
Module xây dựng prompt và xử lý query trước khi đưa vào LLM.

Gồm 2 chức năng chính:
  1. `build_prompt(query, contexts)` : Tạo prompt cuối cùng cho LLM trả lời.
  2. `rewrite_and_classify_query()`  : Viết lại query + phân loại SIMPLE/COMPLEX
                                       trong 1 lần gọi API (tiết kiệm latency).

THAY ĐỔI SO VỚI BẢN CŨ:
  - Trước: `client = genai.Client(...)` được tạo riêng trong file này.
  - Sau: import `client` từ `rag.core.container` — dùng chung 1 client duy nhất.
"""

import json
import os
import re

from rag.core.container import container as _container


def build_prompt(query: str, contexts: list) -> str:
    """
    Tạo prompt hoàn chỉnh để gửi cho LLM sinh câu trả lời cuối cùng.

    Args:
        query    : Câu hỏi (đã được rewrite nếu cần).
        contexts : Danh sách đoạn văn bản context từ retrieval.

    Returns:
        Chuỗi prompt đầy đủ.
    """
    context_text = "\n\n".join([f"{text}" for text in contexts])

    return f"""Bạn là trợ lý tư vấn tuyển sinh của Trường Đại học Bách khoa TP.HCM (HCMUT).

    Thông tin tham khảo:
    {context_text}

    Câu hỏi: {query}

    Hướng dẫn trả lời:
    - Chỉ sử dụng thông tin tham khảo nếu nó trực tiếp trả lời được câu hỏi.
    - Nếu hoàn toàn không có thông tin, trả lời: "Tôi không tìm thấy thông tin này trong cơ sở dữ liệu."
    - TUYỆT ĐỐI Không bịa đặt thông tin.
    - Trả lời ngắn gọn, rõ ràng và trả lời đúng trọng tâm câu hỏi.
    - Nếu câu hỏi hoặc yêu cầu của người dùng không liên quan đến Trường Đại Học Bách Khoa Thành Phố Hồ Chí Minh thì trả lời "Xin lỗi tôi chỉ trả lời những thông tin liên quan đến Trường Đại Học Bách Khoa Thành Phố Hồ Chí Minh, nếu có câu hỏi liên quan đến trường xin hãy cho tôi biết.".
    Trả lời:"""


def rewrite_and_classify_query(current_query: str, history: list) -> tuple[str, str]:
    """
    Args:
        current_query : Câu hỏi hiện tại của người dùng.
        history       : Lịch sử hội thoại từ `rag.chat.history.get_history()`.

    Returns:
        Tuple (rewritten_query: str, query_type: str)
        query_type là "SIMPLE" hoặc "COMPLEX".
    """
    history_text = "Không có"
    if history:
        history_text = "\n\n".join(
            [
                f"User: {turn.get('rewritten', turn['user'])}\nAssistant: {turn['assistant']}"
                for turn in history[-5:]
            ]
        )

    prompt = f"""Dựa trên lịch sử trò chuyện (nếu có), hãy thực hiện 2 nhiệm vụ:
1. Viết lại câu hỏi sau thành câu độc lập, đầy đủ ngữ cảnh (nếu có đại từ chỉ định như 'nó', 'trường này'..., hãy thay thế bằng danh từ cụ thể từ lịch sử). Nếu không cần viết lại, giữ nguyên câu hỏi.
2. Phân loại câu hỏi ĐÃ VIẾT LẠI thành 1 trong 2 loại: SIMPLE (câu hỏi đơn giản/tra cứu thông tin trực tiếp), COMPLEX (câu hỏi phức tạp/cần suy luận/mơ hồ).

Lịch sử trò chuyện:
{history_text}

Câu hỏi hiện tại: "{current_query}"

Bạn PHẢI trả về định dạng JSON chính xác như sau, không xuất thêm bất kỳ chữ nào khác:
{{
  "rewritten_query": "câu hỏi đã viết lại",
  "query_type": "SIMPLE"
}}"""

    response = _container.client.models.generate_content(
        model=os.getenv("model_name"), contents=prompt
    )
    text = response.text.strip()

    if json_match := re.search(r"\{.*\}", text, re.DOTALL):
        text = json_match[0]

    parsed = json.loads(text)
    rewritten = parsed.get("rewritten_query", current_query)
    q_type = parsed.get("query_type", "SIMPLE").upper()
    return rewritten, q_type
