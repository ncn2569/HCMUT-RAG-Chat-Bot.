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
    st.markdown("Agentic RAG của 1 sinh viên năm 4 rảnh rỗi thôi ahihi.")

    if st.button("🔄 Bắt đầu cuộc trò chuyện mới", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Chào bạn! Mình là Nguyen's AI Agent tra cứu thông tin về HCMUT. Mình sẽ tự suy luận để tìm đáp án tốt nhất. Bạn hỏi đi nào, À và tất nhiên là đừng làm khó mình nha mình đần lắm!",
                "avatar": "🤖",
            }
        ]
        agent._pipeline.reset_history()
        st.rerun()

    if st.button("🗑️ Dọn dẹp Semantic Cache", use_container_width=True):
        agent._pipeline.cache.flush()
        st.toast("Đã xóa sạch bộ nhớ đệm!", icon="✅")

    st.divider()
    st.caption("Các tools hiện tại là tra cứu trong database nội bộ (86 câu Q-A), search web và tính điểm đầu vào nhe.")

st.title("🤖 HCMUT Agent Chatbot")
st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
                "content": "Chào bạn! Mình là Nguyen's AI Agent tra cứu thông tin về HCMUT. Mình sẽ tự suy luận để tìm đáp án tốt nhất. Bạn hỏi đi nào, À và tất nhiên là đừng làm khó mình nha mình đần lắm!",
            "avatar": "🤖",
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message.get("avatar")):
        st.markdown(message["content"])
        if "thought_logs" in message and message["thought_logs"]:
            with st.expander("🧠 Trích xuất suy luận & Log hệ thống"):
                st.code(message["thought_logs"], language="text")

if prompt := st.chat_input("Đây là Agentic RAG thiên hướng về việc tra cứu thông tin bạn nhé đừng hỏi gì ảo ma canada quá nha..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.status("Câu này căng như giải tích vậy bạn đợi tớ 1 chút nha...", expanded=True) as status:
            with capture_stdout() as output:
                bot_response = agent.run(query=prompt)

            logs = output.getvalue()

            import re
            
            # Bóc tách phần "thought" từ log để hiển thị như Claude
            thought_text = ""
            rewritten_text = ""
            plan_match = re.search(r'KẾ HOẠCH CỦA LLM \*{15}\n(.*?)\n\*{48}', logs, re.DOTALL)
            if plan_match:
                with contextlib.suppress(Exception):
                    plan_dict = json.loads(plan_match.group(1).strip())
                    thought_text = plan_dict.get("thought", "")
                    rewritten_text = plan_dict.get("rewritten_query", "")

            # Bóc tách nguồn tài liệu (cả DB hoặc bất cứ công cụ nào)
            db_sources = []
            db_match = re.search(r'NGUỒN TÀI LIỆU \*{15}\n(.*?)\n\*{48}', logs, re.DOTALL)
            if db_match:
                with contextlib.suppress(Exception):
                    db_sources = json.loads(db_match.group(1).strip())
            
            # Bóc tách nguồn tài liệu từ Web (Fallback)
            web_sources = []
            web_match = re.search(r'NGUỒN TÀI LIỆU BỔ SUNG \(WEB\) \*{15}\n(.*?)\n\*{48}', logs, re.DOTALL)
            if web_match:
                with contextlib.suppress(Exception):
                    web_sources = json.loads(web_match.group(1).strip())

            if thought_text:
                with st.expander("🤔 Xem logic suy luận của Agent", expanded=True):
                    st.info(thought_text)
                    
            if rewritten_text and rewritten_text != prompt:
                st.info(f"**Câu hỏi đã được hiểu lại:** {rewritten_text}", icon="💡")
                
            if db_sources or web_sources:
                with st.expander("📚 Xem nguồn tài liệu tham khảo"):
                    if db_sources:
                        st.markdown("**Từ Cơ sở dữ liệu/Công cụ (Database/Tools):**")
                        for idx, src in enumerate(db_sources, 1):
                            st.info(f"**[{idx}]** {src}")
                    if web_sources:
                        st.markdown("**Từ Internet (Web Search Fallback):**")
                        for idx, src in enumerate(web_sources, 1):
                            st.success(f"**[{idx}]** {src}")
                            
            if not thought_text and logs.strip() and "NGUỒN TÀI LIỆU" not in logs:
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
