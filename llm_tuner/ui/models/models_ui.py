import gradio as gr

from ...utils.relative_read_file import relative_read_file
from ..css_styles import register_css_style

from .presets.model_presets_ui import model_presets_ui

register_css_style('models', relative_read_file(__file__, "style.css"))


def models_ui():
    things_that_might_hang = []
    with gr.Blocks() as models_ui_blocks:
        with gr.Tab("Presets"):
            model_presets_ui()
    models_ui_blocks.load(_js=relative_read_file(__file__, "script.js"))
