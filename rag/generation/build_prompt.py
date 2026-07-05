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


def build_prompt(query: str, contexts: list, history_text: str = "") -> str:
    """
    Tạo prompt hoàn chỉnh để gửi cho LLM sinh câu trả lời cuối cùng.

    Args:
        query        : Câu hỏi (đã được rewrite nếu cần).
        contexts     : Danh sách đoạn văn bản context từ retrieval.
        history_text : Lịch sử hội thoại (nếu có).

    Returns:
        Chuỗi prompt đầy đủ.
    """
    context_text = "\n\n".join([f"{text}" for text in contexts])

    history_section = (
        f"\n    Lịch sử trò chuyện:\n    {history_text}\n" if history_text else ""
    )

    return f"""Bạn là một Chuyên viên Tư vấn Tuyển sinh chuyên nghiệp, thân thiện và nhiệt huyết của Trường Đại học Bách khoa - ĐHQG TP.HCM (HCMUT).

    DƯỚI ĐÂY LÀ THÔNG TIN TỪ CƠ SỞ DỮ LIỆU & WEB:
    {context_text}{history_section}

    CÂU HỎI CỦA SINH VIÊN: {query}

    HƯỚNG DẪN TRẢ LỜI (TUYỆT ĐỐI TUÂN THỦ):
    1. CHIẾT XUẤT THÔNG TIN: Chỉ sử dụng thông tin từ mục "THÔNG TIN TỪ CƠ SỞ DỮ LIỆU & WEB" ở trên. TUYỆT ĐỐI không bịa đặt hoặc tự chém gió thêm số liệu, quy chế, hay năm học nếu không có trong dữ liệu.
    2. CHỐNG ẢO GIÁC (HALLUCINATION): TUYỆT ĐỐI không tự bịa số liệu. Nếu dữ liệu HOÀN TOÀN KHÔNG có bất cứ thông tin nào trả lời được câu hỏi, BẮT BUỘC bạn phải in ra ĐÚNG 1 CÂU SAU (không in thêm gì khác): "Tôi không tìm thấy thông tin này trong cơ sở dữ liệu." Nếu dữ liệu chỉ giải đáp được MỘT PHẦN câu hỏi (ví dụ hỏi 2 ngành nhưng chỉ có data 1 ngành), hãy trả lời phần có dữ liệu và báo rõ phần nào không có thông tin.
    3. PHẠM VI: Nếu người dùng hỏi các vấn đề hoàn toàn không liên quan đến Bách Khoa TP.HCM (HCMUT) hay tuyển sinh đại học (ví dụ: công thức nấu ăn, dự báo thời tiết, kiến thức phổ thông ngoài lề), hãy lịch sự từ chối: "Xin lỗi, tôi chỉ được huấn luyện để giải đáp các thông tin liên quan đến Trường Đại Học Bách Khoa TP.HCM. Bạn có thắc mắc gì về tuyển sinh không ạ?"
    4. HÀNH VĂN: Trình bày rõ ràng, mạch lạc, dùng gạch đầu dòng nếu có liệt kê. Giọng điệu chuyên nghiệp, xưng "Tôi" hoặc "Trường" và gọi người hỏi là "bạn" hoặc "em".

    CÂU TRẢ LỜI CỦA BẠN:"""


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
