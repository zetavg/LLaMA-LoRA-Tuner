import os
import sys

import fire
import gradio as gr

from llama_lora.globals import Global
from llama_lora.models import prepare_base_model
from llama_lora.ui.main_page import main_page, get_page_title, main_page_custom_css
from llama_lora.utils.data import init_data_dir



def main(
    base_model: str = "",
    data_dir: str = "",
    base_model_choices: str = "",
    trust_remote_code: bool = False,
    # Allows to listen on all interfaces by providing '0.0.0.0'.
    server_name: str = "127.0.0.1",
    share: bool = False,
    skip_loading_base_model: bool = False,
    load_8bit: bool = False,
    ui_show_sys_info: bool = True,
    ui_dev_mode: bool = False,
    wandb_api_key: str = "",
    wandb_project: str = "",
):
    '''
    Start the LLaMA-LoRA Tuner UI.

    :param base_model: (required) The name of the default base model to use.
    :param data_dir: (required) The path to the directory to store data.

    :param base_model_choices: Base model selections to display on the UI, seperated by ",". For example: 'decapoda-research/llama-7b-hf,nomic-ai/gpt4all-j'.

    :param server_name: Allows to listen on all interfaces by providing '0.0.0.0'.
    :param share: Create a public Gradio URL.

    :param wandb_api_key: The API key for Weights & Biases. Setting either this or `wandb_project` will enable Weights & Biases.
    :param wandb_project: The default project name for Weights & Biases. Setting either this or `wandb_api_key` will enable Weights & Biases.
    '''

    base_model = base_model or os.environ.get("LLAMA_LORA_BASE_MODEL", "")
    data_dir = data_dir or os.environ.get("LLAMA_LORA_DATA_DIR", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    assert (
        data_dir
    ), "Please specify a --data_dir, e.g. --data_dir='./data'"

    Global.default_base_model_name = Global.base_model_name = base_model

    if base_model_choices:
        base_model_choices = base_model_choices.split(',')
        base_model_choices = [name.strip() for name in base_model_choices]
        Global.base_model_choices = base_model_choices

    if base_model not in Global.base_model_choices:
        Global.base_model_choices = [base_model] + Global.base_model_choices

    Global.trust_remote_code = trust_remote_code

    Global.data_dir = os.path.abspath(data_dir)
    Global.load_8bit = load_8bit

    if len(wandb_api_key) > 0:
        Global.enable_wandb = True
        Global.wandb_api_key = wandb_api_key
    if len(wandb_project) > 0:
        Global.enable_wandb = True
        Global.wandb_project = wandb_project

    Global.ui_dev_mode = ui_dev_mode
    Global.ui_show_sys_info = ui_show_sys_info

    os.makedirs(data_dir, exist_ok=True)
    init_data_dir()

    if (not skip_loading_base_model) and (not ui_dev_mode):
        prepare_base_model(base_model)

    with gr.Blocks(title=get_page_title(), css=main_page_custom_css()) as demo:
        main_page()

    demo.queue(concurrency_count=1).launch(server_name=server_name, share=share)


if __name__ == "__main__":
    fire.Fire(main)
