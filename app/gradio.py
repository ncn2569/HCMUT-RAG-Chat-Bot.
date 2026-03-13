import gradio as gr
from rag.pipeline import rag_query, reset_history

def respond(message, chat_history):
    """
    Xử lý tin nhắn và cập nhật chat history.
    chat_history là list of dicts với role và content.
    """
    chat_history = chat_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": "⏳ Đang xử lý..."}
    ]
    yield "", chat_history

    bot_message = rag_query(message)

    chat_history[-1] = {"role": "assistant", "content": bot_message}
    yield "", chat_history

def clear_chat():
    """Xóa lịch sử chat trong cả Gradio và internal history"""
    reset_history()
    return []  

with gr.Blocks(
    title="🎓 HCMUT Chatbot"
) as demo:
    gr.Markdown("""
    # 🎓 HCMUT Chatbot
    **Trợ lý tư vấn tuyển sinh** · Trường Đại học Bách khoa TP.HCM
    ---
    """)

    chatbot = gr.Chatbot(
        elem_id="chatbot",
        show_label=False,
        placeholder="Hãy đặt câu hỏi về tuyển sinh HCMUT...",
        layout="bubble",
    )

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Nhập câu hỏi của bạn...",
            show_label=False,
            container=False,
            scale=9,
            autofocus=True,
        )
        send = gr.Button("Gửi", elem_id="send-btn", scale=1, min_width=80)

    with gr.Row():
        clear = gr.Button("🔄 Xóa lịch sử", elem_id="clear-btn", size="sm")

    gr.Markdown("""
    <div style='text-align:center; color:#9ca3af; font-size:12px; margin-top:8px'>
    Lưu ý: Data là bản thân tự crawl nên có thể còn hơn ít và hạn chế.
    </div>
    """)

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    send.click(respond, [msg, chatbot], [msg, chatbot])
    clear.click(clear_chat, None, chatbot)