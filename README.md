---
title: HCMUT Chatbot
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
---
# 🎓 HCMUT RAG Chatbot

Trợ lý tư vấn tuyển sinh thông minh cho Trường Đại học Bách khoa TP.HCM, sử dụng RAG (Retrieval-Augmented Generation) với Google Gemini.

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue.svg)

## 📝 Tâm sự của tác giả 
- Dữ liệu hiện tại là từ một cá nhân rảnh rỗi iu trường (optional + seasonal) và có đam mê với AI. 
- Dữ liệu theo format Q-A, với 1 cột Question ghi các câu hỏi và 1 cột Answers ghi các câu trả lời (đây là dạng dễ crawl nhất). Tổng tất cả là 80 câu dòng (Tớ biết nó ít nhưng dòng nào cũng chan chứa mồ hôi hết).
- Cần API key Gemini để chạy (của tớ lúc làm thì tất nhiên là free rồi, svien bkhoa nghèo có tiếng mà).
- File .env và thư mục data/raw và data/processed đã được ignore nhưng vẫn còn data/vectors nếu bạn múa ngó thử về data mình đã cất công hái lượm.
- Tất nhiên là tương lai mình sẽ phát triển lên thêm (possibly agents nếu mình vô tình lượm đc một cái api key unlimited còn hiện tại thì chỉ là một Naive RAG bình thường thôi)

## 🚀 Tính năng

- **RAG Pipeline**: Kết hợp Dense Search + Sparse Search + HYDE + RRF Fusion + Reranker.
- **Two-Stage Semantic Cache**: Tối ưu tốc độ cực sướng (< 0.5s) bằng kiến trúc Cache 2 tầng (Vector Search + Cross-Encoder Reranker) và lưu trữ bất đồng bộ (Asynchronous).
- **Query Rewriting & Routing**: Gộp chung bước phân loại câu hỏi (Simple/Complex) và viết lại câu hỏi vào 1 lượt gọi API duy nhất, giảm triệt để độ trễ.
- **Chat History**: Quản lý ngữ cảnh hội thoại đa lượt.
- **Gemini Integration**: Sử dụng Gemma 4 31B cho generation & Gemini Embedding 001 cho retrieval.

## 📊 Đánh giá hệ thống (RAGAS Evaluation Metrics)

Hệ thống được đánh giá tự động thông qua framework **RAGAS**, các chỉ số gồm: (context precision + faithfulness + context recall + answer relevance).

Chấm bằng gemini 3 flash lite, thằng này thì nó đần hơn gemma 4 31b mình dùng để trả lời nhưng mà nó là con free duy nhất mà mình chạy mượt với ragas, mấy thằng khác thì nó không hỗ trợ hoặc lâu vl hoặc trả phí.

| Sample User Query | Faithfulness | Context Recall | Answer Relevancy | Context Precision |
| :--- | :---: | :---: | :---: | :---: |
| *Mã trường khi em đăng ký xét tuyển vào Bách khoa là gì?* | 1.0 | 1.0 | 0.7997 | 1 |
| *Em thuộc diện hộ nghèo là người dân tộc thiểu số, vậy có được miễn giảm học phí không và nộp hồ sơ ở đâu?* | 1.0 | 1.0 | 0.8669 | 1.0 |
| *Khoa Kỹ thuật Giao thông đào tạo những ngành nào vậy ạ?* | 1.0 | 1.0 | 0.8576 | 1.0 |
| *Ngành Khoa học Máy tính học những môn cơ sở ngành gì và sau khi ra trường có thể làm ở đâu?* | 1.0 | 1.0 | 0.8582 | 1.0 |
## 📁 Cấu trúc dự án
```text
hcmut-rag-chatbot/
├── app/
│   └── streamlit_app.py       # Giao diện web 
├── rag/
│   ├── ingestion/
│   │   └── chunking.py        # Xử lý chunking dữ liệu
│   ├── embedding/
│   │   └── embed.py           # Embedding với Gemini API
│   ├── retrieval/
│   │   ├── hyde.py            # Hypothetical Document Embedding
│   │   ├── dense_search.py    # Dense vector search
│   │   ├── bm25.py            # Sparse search using bm25
│   │   ├── rerank.py          # Reranker with bge-reranker-v2-m3
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
├── main.py                    
├── evaluation.py              # siêu phiền siêu lâu nếu không sở hữu api key có tiền (nghèo mà chịu thôi)   
├── report5.xlsx               # này là file đánh giá hiện tại.
├── Dockerfile                 
└── requirements.txt           
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
model_name="gemma-4-31b-it" * gg mới update
model_embedding_name="gemini-embedding-001"

# Hugging Fac
HF_HOME=đường dẫn đến cache hugging face của bạn nếu không có thì sẽ dùng default.(mình thì ít ổ C nên cài vào ổ D)
HF_TOKEN= your huggings face token.
```
🔑 Lấy API key miễn phí tại: Google AI Studio

### 5. Chuẩn bị dữ liệu

**Bước 5.1:** Đặt file Excel vào thư mục `data/raw/`

**Bước 5.2:** Mở file `main.py`, sửa như sau để chạy embedding nếu có data mới:
```python
    # Trong main.py
    if __name__ == "__main__":
        from rag.embedding.embed import embedding
        embedding()
        # subprocess.run([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"])
```
**Bước 5.3:** Chạy lệnh để tạo vector embeddings:
```bash
    python main.py
```
**Bước 5.4:** Sau khi chạy xong, sửa lại `main.py` để chạy web:
```python
    # Trong main.py
    if __name__ == "__main__":
        from rag.embedding.embed import embedding
        #embedding()
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"])
```
**Bước 5.5:** Giờ chạy chatbot:
```bash
    python main.py
```
```text
⚙️ Cách hoạt động (Pipeline)
User Query 
    → Đi qua Semantic Cache (Vector Search + Reranker). Nếu Hit -> Trả kết quả ngay (0.5s).
    → Nếu Miss -> Query Rewriting & Routing (Gộp 2 bước vào 1 API call).
    → (Nếu Complex) HYDE (tạo hypothetical query).
    → Dense Search + Sparse Search (bm25).
    → Weighted RRF Fusion (k=60) -> Top 15. 
    → Reranking -> Top 5 final.
    → Build Prompt & Generate Answer.
    → Lưu ngầm câu trả lời vào Semantic Cache (Asynchronous) & Cập nhật History.
```

---

## 🤖 HCMUT Agentic RAG (Chế độ Agent)

Ngoài RAG pipeline cơ bản, project đã được nâng cấp lên chuẩn **Agentic RAG** với kiến trúc Plan-and-Execute + Self-Reflection. Đây là chế độ đang được triển khai thực tế qua `app/streamlit_agent.py`.

### Tính năng nổi bật của Agent

- **Plan-and-Execute:** LLM tự lập kế hoạch — tự suy luận, tự chọn Tool (`search_db`, `search_web`, hoặc `predict_admission`), tự sinh `rewritten_query` — tất cả trong **1 API call duy nhất**.
- **Context-Aware History (5 lượt):** Agent nhớ 5 lượt hội thoại gần nhất. Nếu user hỏi "Học phí của nó?", Agent tự hiểu "nó" là ngành vừa đề cập ở lượt trước và viết lại câu hỏi hoàn chỉnh trước khi tìm kiếm.
- **Dự đoán cơ hội đỗ (Predict Admission):** Tích hợp công cụ tính điểm quy đổi (ĐGNL, THPT, Học bạ) và đối khớp mờ tên ngành (Fuzzy Match) để so sánh với phổ điểm các năm và đưa ra đánh giá cơ hội đậu.
- **Self-Reflection & Fallback:** Nếu DB nội bộ không đủ thông tin, Agent tự phát hiện và tự động fallback sang tìm kiếm Web (Tavily API) mà không cần người dùng can thiệp.
- **Semantic Cache tích hợp:** Câu hỏi trùng lặp được trả về từ Cache với **0 LLM API call**, tiết kiệm triệt để giới hạn RPM.
- **Kháng lỗi:** Tất cả điểm gọi API đều có `try/except`, bot không crash khi API lỗi hoặc hết quota.

### Cấu trúc file bổ sung (Agent Mode)

```text
hcmut-rag-chatbot/
├── app/
│   ├── streamlit_app.py       # Giao diện RAG thuần (pipeline cũ)
│   └── streamlit_agent.py     # Giao diện Agent (đang dùng)
├── rag/
│   ├── agent.py               # RAGAgent — Plan-and-Execute core
│   └── chat/
│       ├── history.py         # Quản lý lịch sử hội thoại
│       └── semantic_cache.py  # Cache ngữ nghĩa 2 tầng
└── agent_architecture.md      # Tài liệu kiến trúc Agent chi tiết
```

### Số lượng API calls per lượt hỏi

| Kịch bản | LLM Calls | Embedding | Tavily | Tổng RPM |
| :--- | :---: | :---: | :---: | :---: |
| Cache Hit | **0** | 1 | 0 | **1** |
| DB có đáp án | **2** | 1 | 0 | **3** |
| Phải Fallback Web | **3** | 1 | 1 | **4** |

### Chạy ở chế độ Agent

```bash
python -m streamlit run app/streamlit_agent.py
```

> Xem chi tiết kiến trúc tại [agent_architecture.md](agent_architecture.md).

### Thêm biến môi trường cho Agent

```env
tavily_key="your-tavily-api-key"   # Lấy miễn phí tại tavily.com
```