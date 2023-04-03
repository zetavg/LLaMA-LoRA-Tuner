import gradio as gr

from ..globals import Global

from .inference_tab import inference_tab


def main_page():
    with gr.Blocks(
            title="LLaMA-LoRA",
            css="") as demo:
        gr.Markdown(f"""
            # {Global.ui_title}

            Hello world!
            """)
        inference_tab()
        if Global.ui_show_sys_info:
            gr.Markdown(f"""
                <small>Data dir: `{Global.data_dir}`</small>
                """)
