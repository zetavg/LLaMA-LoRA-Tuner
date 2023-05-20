import gradio as gr

from ...utils.relative_read_file import relative_read_file
from ..css_styles import register_css_style

register_css_style('models', relative_read_file(__file__, "style.css"))


def models_ui():
    with gr.Blocks() as models_ui_blocks:
        gr.Markdown("This is models_ui", elem_id="models_ui")

    models_ui_blocks.load(_js=relative_read_file(__file__, "script.js"))
