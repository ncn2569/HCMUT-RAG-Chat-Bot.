from google import genai
from dotenv import load_dotenv
import os
load_dotenv('config/.env')
client = genai.Client(api_key=os.getenv('API_KEY'))
import json
data = []
with open('data/vectors/vectors1.jsonl', "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            data.append(json.loads(line))  # ✅
def build_prompt(query: str, contexts: list) -> str:
    """
    Build prompt với Role + Context + Constraints
    """
    # Gộp top 3 QA thành context
    context_text = "\n\n".join([
        f"{text}"
        for text in contexts  
    ])

    prompt = f"""Bạn là trợ lý tư vấn tuyển sinh của Trường Đại học Bách khoa TP.HCM (HCMUT).

    Thông tin tham khảo:
    {context_text}

    Câu hỏi: {query}

    Hướng dẫn trả lời:
    - Chỉ sử dụng thông tin tham khảo nếu nó **trực tiếp** trả lời được câu hỏi.
    - Nếu thông tin tham khảo không liên quan đến câu hỏi thì không cần dùng cũng được.
    - Nếu hoàn toàn không có thông tin, trả lời: "Tôi không tìm thấy thông tin này trong cơ sở dữ liệu."
    - Không bịa đặt thông tin.
    - Trả lời ngắn gọn, rõ ràng và trả lời đúng trọng tâm câu hỏi.
    - Không đặt thêm câu hỏi, chỉ trả lời một lần.

    Trả lời:"""

    return prompt

def rewrite_query_with_full_history(current_query: str, history: list) -> str:
    """
    Viết lại query dựa trên toàn bộ history để tạo standalone query
    """
    if not history:
        return current_query

    # Toàn bộ history
    history_text = "\n\n".join([
        f"User: {turn.get('rewritten', turn['user'])}\nAssistant: {turn['assistant']}"
        for turn in history[-5:]  # ✅ dùng rewritten nếu có, fallback về user gốc
    ])

    rewrite_prompt = f"""Dựa trên lịch sử trò chuyện gần nhất, viết lại câu hỏi sau thành câu độc lập, đầy đủ ngữ cảnh.

    Lịch sử trò chuyện:
    {history_text}

    Câu hỏi mới: "{current_query}"

    Hướng dẫn:
    - Nếu câu hỏi có đại từ hoặc thiếu ngữ cảnh, bổ sung từ lịch sử
    - Giữ nguyên ý nghĩa gốc, không thay đổi ý nghĩa của câu hỏi,
    - Chỉ viết lại câu hỏi, TUYỆT ĐỐI KHÔNG trả lời.

    Câu hỏi đã viết lại:"""

    response = client.models.generate_content(
        model=os.getenv('model_name'),
        contents=rewrite_prompt
    )

    rewritten = response.text.strip() if response.text else current_query
    rewritten = rewritten.strip('"').strip("'")

    return rewritten