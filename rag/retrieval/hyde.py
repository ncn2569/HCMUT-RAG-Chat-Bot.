import re
from google import genai
from dotenv import load_dotenv
import os
load_dotenv('config/.env')
client = genai.Client(api_key=os.getenv('API_KEY'))
def generate_hypothetical_query(query, model_name=os.getenv('model_name')):

    prompt = f"""Bạn đang hỗ trợ hệ thống tìm kiếm trong cơ sở dữ liệu hỏi đáp.

    Hãy viết lại câu sau bằng cách diễn đạt khác nhưng giữ nguyên ý nghĩa.
    Mục tiêu là tạo một câu hỏi tương tự về mặt ngữ nghĩa để giúp tìm được các câu hỏi và tài liệu liên quan.

    Hướng dẫn:
    - Diễn đạt lại tự nhiên bằng cách khác.
    - Có thể thay đổi từ ngữ hoặc cấu trúc câu.
    - Chỉ trả về câu hỏi đã viết lại, không giải thích.

    Câu hỏi gốc: {query}

    Câu hỏi tương tự:"""

    response = client.models.generate_content(
        model=os.getenv('model_name'),
        contents=prompt
    )

    hypothetical = response.text.strip() if response.text else query

    # Clean: lấy dòng đầu, bỏ prefix nếu có
    hypothetical = hypothetical.split("\n")[0].strip()
    hypothetical = re.sub(r'^(Câu hỏi tương tự[:：]\s*)', '', hypothetical)


    return hypothetical

