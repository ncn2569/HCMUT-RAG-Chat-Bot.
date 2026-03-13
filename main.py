from dotenv import load_dotenv
import os
import json
import numpy as np 
from rag.embedding.embed import embedding
load_dotenv('config/.env')
os.environ["HF_HOME"] = os.getenv("HF_HOME")

from app.gradio import demo
import gradio as gr
if __name__ == "__main__":
    # embedding()
    demo.launch(
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
        ),
        css="""
        #chatbot { height: calc(100vh - 160px); }
        .gradio-container { max-width: 100% !important; margin: 0; padding: 0 20px; }
        footer { display: none !important; }
        #send-btn { background: #1a56db; color: white; }
        #clear-btn { border: 1px solid #e5e7eb; }
        """
    )