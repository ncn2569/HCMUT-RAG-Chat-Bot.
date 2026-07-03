"""
app/streamlit_app.py
====================
UI Streamlit cho RAG Pipeline (chế độ không dùng Agent).
"""

import streamlit as st

from rag.core.container import container
from rag.pipeline import RAGPipeline

# Khởi tạo pipeline 1 lần duy nhất, cache lại trong session Streamlit
# st.cache_resource đảm bảo object nặng không bị tạo lại mỗi lần user tương tác
@st.cache_resource
def get_pipeline() -> RAGPipeline:
    return RAGPipeline(container)


pipeline = get_pipeline()

st.set_page_config(page_title="HCMUT-NCN-Chatbot", page_icon="🎓", layout="centered")

with st.sidebar:
    st.title("⚙️ Config zone")
    st.markdown(
        "Trợ lý ảo(hơi cùi) hỗ trợ giải đáp thắc mắc về trường ĐH Bách khoa TP.HCM."
    )

    if st.button("🔄 Bắt đầu cuộc trò chuyện mới", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Chào bạn! Mình là trợ lý ảo của HCMUT. Mình có thể giúp gì cho bạn hôm nay?",
                "avatar": "🎓",
            }
        ]
        pipeline.reset_history()
        st.rerun()

    if st.button("🗑️ Dọn dẹp Semantic Cache", use_container_width=True):
        pipeline.cache.flush()
        st.toast("Đã xóa sạch bộ nhớ đệm!", icon="✅")

    st.divider()
    st.caption(
        "Lưu ý: Data là bản thân mình tự crawl nên có thể còn hơi ít và hạn chế "
        "nhưng đảm bảo 80 dòng này dòng nào cũng tràn đầy tâm huyết."
    )

st.title("🎓 HCMUT-NCN-Chatbot")
st.markdown(
    "Trợ lý tư vấn RAG với tập dữ liệu homemade bởi 1 sinh viên năm 3 rảnh rỗi."
)
st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Chào bạn! Mình là trợ lý ảo của HCMUT. Mình có thể giúp gì cho bạn hôm nay?",
            "avatar": "🎓",
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message.get("avatar")):
        st.markdown(message["content"])

if prompt := st.chat_input("Hãy đặt câu hỏi về trường iu HCMUT của mình đi..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("Đang RAGGING làm ơn hãy đợi 1 xí xíu"):
            bot_response = pipeline.query(prompt)
            st.markdown(bot_response)

    st.session_state.messages.append({"role": "assistant", "content": bot_response})
