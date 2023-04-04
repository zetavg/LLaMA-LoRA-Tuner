import os
import sys

import fire
import gradio as gr

from llama_lora.globals import Global
from llama_lora.ui.main_page import main_page
from llama_lora.utils.data import init_data_dir


def main(
    load_8bit: bool = False,
    base_model: str = "",
    data_dir: str = "",
    # Allows to listen on all interfaces by providing '0.0.0.0'.
    server_name: str = "127.0.0.1",
    share: bool = False,
    ui_show_sys_info: bool = True,
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    assert (
        data_dir
    ), "Please specify a --data_dir, e.g. --data_dir='./data'"

    Global.base_model = base_model
    Global.data_dir = data_dir
    Global.load_8bit = load_8bit

    Global.ui_show_sys_info = ui_show_sys_info

    os.makedirs(data_dir, exist_ok=True)
    init_data_dir()

    with gr.Blocks(title=Global.ui_title) as demo:
        main_page()

    demo.queue().launch(server_name=server_name, share=share)


if __name__ == "__main__":
    fire.Fire(main)
