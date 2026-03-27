# 🎓 HCMUT RAG Chatbot

Trợ lý tư vấn tuyển sinh thông minh cho Trường Đại học Bách khoa TP.HCM, sử dụng RAG (Retrieval-Augmented Generation) với Google Gemini.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)

## 🚀 Tính năng

- **RAG Pipeline**: Kết hợp Dense Search + HYDE + RRF Fusion
- **Chat History**: Quản lý ngữ cảnh hội thoại đa lượt
- **Query Rewriting**: Tự động viết lại câu hỏi dựa trên lịch sử
- **Gemini Integration**: Sử dụng Gemma 3 27B cho generation & Gemini Embedding cho retrieval

## 📁 Cấu trúc dự án
```text
hcmut-rag-chatbot/
├── app/
│   └── gradio.py              # Giao diện web Gradio
├── rag/
│   ├── ingestion/
│   │   └── chunking.py        # Xử lý chunking dữ liệu
│   ├── embedding/
│   │   └── embed.py           # Embedding với Gemini API
│   ├── retrieval/
│   │   ├── hyde.py            # Hypothetical Document Embedding
│   │   ├── dense_search.py    # Dense vector search
│   │   └── rrf_fuse.py        # Reciprocal Rank Fusion
│   ├── generation/
│   │   └── build_prompt.py    # Prompt engineering & query rewriting
│   ├── chat/
│   │   └── history.py         # Quản lý lịch sử hội thoại
│   └── pipeline.py            # RAG pipeline chính
├── data/
│   ├── raw/                   # Dữ liệu gốc (Excel)
│   ├── processed/             # Dữ liệu đã xử lý (JSONL)
│   └── vectors/               # Vector embeddings (NPY)
├── config/
│   └── .env                   # API keys (tự tạo với API_KEY theo template)
├── main.py                    # Entry point
└── requirements.txt           # Dependencies
```
## 🛠️ Cài đặt

### 1. Clone repository
```bash
git clone https://github.com/username/hcmut-rag-chatbot.git
cd hcmut-rag-chatbot
```
### 2. Tạo môi trường ảo
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```
### 3. Cài đặt dependencies
```bash
pip install -r requirements.txt
```
### 4. Cấu hình biến môi trường

Tạo file `config/.env` với nội dung:

```env
# Google Gemini API
API_KEY="your-gemini-api-key-here"
model_name="gemma-3-27b-it"
model_embedding_name="gemini-embedding-001"

# Hugging Face (optional)
HF_HOME=đường dẫn đến cache hugging face của bạn nếu không có thì sẽ dùng default.
HF_TOKEN= your huggings face token.
```
🔑 Lấy API key miễn phí tại: Google AI Studio

### 5. Chuẩn bị dữ liệu

**Bước 5.1:** Đặt file Excel vào thư mục `data/raw/`

**Bước 5.2:** Mở file `main.py`, sửa như sau để chạy embedding:

    # Comment dòng này lại
    # demo.launch(...)
    
    # Uncomment dòng này
    embedding()

**Bước 5.3:** Chạy lệnh để tạo vector embeddings:

    python main.py

**Bước 5.4:** Sau khi chạy xong, sửa lại `main.py` để chạy web:

    # Uncomment dòng này
    demo.launch(...)
    
    # Comment dòng này lại
    # embedding()

**Bước 5.5:** Giờ chạy chatbot:

    python main.py
```text
🔧 Cách hoạt động (Pipeline)
User Query → Query Rewriting (dựa trên history) 
    → HYDE (tạo hypothetical query)
    → Dense Search (2 queries: original + hyde)
    → RRF Fusion (k=30)
    → Top 10 candidates → Top 3 final
    → Build Prompt 
    → Generate Answer
    → Update History
```

## 📝 Lưu ý
- Dữ liệu hiện tại là bản crawl thử nghiệm, có thể chưa đầy đủ.
- Dữ liệu theo format Q-A, với 1 cột Question ghi các câu hỏi và 1 cột Answers ghi các câu trả lời.
- Cần API key Gemini để chạy (có free tier).
- File .env và thư mục data/ đã được ignore để bảo mật.
