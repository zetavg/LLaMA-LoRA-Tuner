import gradio as gr

from .tokenizer_ui import tokenizer_ui


def tools_ui():
    with gr.Tab(label="Tokenizer"):
        tokenizer_ui()
