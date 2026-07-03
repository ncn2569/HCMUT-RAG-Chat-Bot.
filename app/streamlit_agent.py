"""
app/streamlit_agent.py
======================
UI Streamlit cho RAGAgent (chế độ Plan-and-Execute Agent).
"""

import contextlib
import io
import json
import sys

import streamlit as st

from rag.agent import RAGAgent
from rag.core.container import container
from rag.pipeline import RAGPipeline


@st.cache_resource
def get_agent() -> RAGAgent:
    """Khởi tạo pipeline và agent 1 lần duy nhất, cache lại trong session."""
    pipeline = RAGPipeline(container)
    return RAGAgent(pipeline, container)


agent = get_agent()


@contextlib.contextmanager
def capture_stdout():
    """Context manager bắt toàn bộ lệnh print() trong agent để hiện lên UI."""
    old_stdout = sys.stdout
    captured = io.StringIO()
    sys.stdout = captured
    try:
        yield captured
    finally:
        sys.stdout = old_stdout


st.set_page_config(
    page_title="HCMUT Agent Chatbot", page_icon="🤖", layout="centered"
)

with st.sidebar:
    st.title("⚙️ Config zone (Agent Mode)")
    st.markdown("Giao diện thử nghiệm luồng **Plan-and-Execute Agent**.")

    if st.button("🔄 Bắt đầu cuộc trò chuyện mới", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Chào bạn! Mình là AI Agent của HCMUT. Mình sẽ tự suy luận để tìm đáp án tốt nhất. Bạn hỏi đi nào!",
                "avatar": "🤖",
            }
        ]
        agent._pipeline.reset_history()
        st.rerun()

    if st.button("🗑️ Dọn dẹp Semantic Cache", use_container_width=True):
        agent._pipeline.cache.flush()
        st.toast("Đã xóa sạch bộ nhớ đệm!", icon="✅")

    st.divider()
    st.caption("Phiên bản chạy bằng LLM Routing, tự quyết định dùng DB hay gọi API Web.")

st.title("🤖 HCMUT Agent Chatbot")
st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Chào bạn! Mình là AI Agent của HCMUT. Mình sẽ tự suy luận để tìm đáp án tốt nhất. Bạn hỏi đi nào!",
            "avatar": "🤖",
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message.get("avatar")):
        st.markdown(message["content"])
        if "thought_logs" in message and message["thought_logs"]:
            with st.expander("🧠 Xem quá trình suy luận (Plan & Tool)"):
                st.code(message["thought_logs"], language="json")

if prompt := st.chat_input("Hỏi thử 1 câu đánh đố xem Agent xử lý sao nhé..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.status("Đang suy nghĩ...", expanded=True) as status:
            with capture_stdout() as output:
                bot_response = agent.run(query=prompt, history=[])

            logs = output.getvalue()

            # Bóc tách phần "thought" từ log để hiển thị như Claude
            thought_text = ""
            if "KẾ HOẠCH CỦA LLM" in logs:
                with contextlib.suppress(Exception):
                    json_str = (
                        logs.split("KẾ HOẠCH CỦA LLM")[1]
                        .split("************************************************")[0]
                        .strip()
                        .strip("*")
                        .strip()
                    )
                    plan_dict = json.loads(json_str)
                    thought_text = plan_dict.get("thought", "")

            if thought_text:
                st.markdown(f"_{thought_text}_")
            elif logs.strip():
                st.code(logs, language="text")

            status.update(label="Suy nghĩ xong", state="complete", expanded=False)

        st.markdown(bot_response)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": bot_response,
                "thought_logs": logs,
                "avatar": "🤖",
            }
        )
