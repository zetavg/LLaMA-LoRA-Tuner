from typing import Union

import gradio as gr
import fire
import os
import yaml

from llama_lora.config import Config, process_config
from llama_lora.globals import initialize_global
from llama_lora.utils.data import init_data_dir
from llama_lora.models import prepare_base_model
from llama_lora.ui.main_page import (
    main_page, get_page_title
)
from llama_lora.ui.css_styles import get_css_styles


def main(
    base_model: Union[str, None] = None,
    data_dir: Union[str, None] = None,
    base_model_choices: Union[str, None] = None,
    trust_remote_code: Union[bool, None] = None,
    server_name: str = "127.0.0.1",
    share: bool = False,
    skip_loading_base_model: bool = False,
    auth: Union[str, None] = None,
    load_8bit: Union[bool, None] = None,
    ui_show_sys_info: Union[bool, None] = None,
    ui_dev_mode: Union[bool, None] = None,
    wandb_api_key: Union[str, None] = None,
    wandb_project: Union[str, None] = None,
    hf_access_token: Union[str, None] = None,
    timezone: Union[str, None] = None,
    config: Union[str, None] = None,
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

    :param hf_access_token: Provide an access token to load private models form Hugging Face Hub. An access token can be created at https://huggingface.co/settings/tokens.
    '''

    config_from_file = read_yaml_config(config_path=config)
    if config_from_file:
        for key, value in config_from_file.items():
            if key == "server_name":
                server_name = value
                continue
            if not hasattr(Config, key):
                available_keys = [k for k in vars(
                    Config) if not k.startswith('__')]
                raise ValueError(
                    f"Invalid config key '{key}' in config.yaml. Available keys: {', '.join(available_keys)}")
            setattr(Config, key, value)

    if base_model is not None:
        Config.default_base_model_name = base_model

    if base_model_choices is not None:
        Config.base_model_choices = base_model_choices

    if trust_remote_code is not None:
        Config.trust_remote_code = trust_remote_code

    if data_dir is not None:
        Config.data_dir = data_dir

    if load_8bit is not None:
        Config.load_8bit = load_8bit

    if auth is not None:
        try:
            [Config.auth_username, Config.auth_password] = auth.split(':')
        except ValueError:
            raise ValueError("--auth must be in the format <username>:<password>, e.g.: --auth='username:password'")

    if hf_access_token is not None:
        Config.hf_access_token = hf_access_token

    if wandb_api_key is not None:
        Config.wandb_api_key = wandb_api_key

    if wandb_project is not None:
        Config.default_wandb_project = wandb_project

    if timezone is not None:
        Config.timezone = timezone

    if ui_dev_mode is not None:
        Config.ui_dev_mode = ui_dev_mode

    if ui_show_sys_info is not None:
        Config.ui_show_sys_info = ui_show_sys_info

    process_config()
    initialize_global()

    assert (
        Config.default_base_model_name
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    assert (
        Config.data_dir
    ), "Please specify a --data_dir, e.g. --data_dir='./data'"

    init_data_dir()

    if (not skip_loading_base_model) and (not Config.ui_dev_mode):
        prepare_base_model(Config.default_base_model_name)

    with gr.Blocks(title=get_page_title(), css=get_css_styles()) as demo:
        main_page()

    demo.queue(concurrency_count=1).launch(
        server_name=server_name,
        share=share,
        auth=((Config.auth_username, Config.auth_password)
              if Config.auth_username and Config.auth_password else None)
    )


def read_yaml_config(config_path: Union[str, None] = None):
    if not config_path:
        app_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(app_dir, 'config.yaml')

    if not os.path.exists(config_path):
        return None

    print(f"Loading config from {config_path}...")
    with open(config_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


if __name__ == "__main__":
    fire.Fire(main)
elif __name__ == "app":  # running in gradio reload mode (`gradio`)
    try:
        main()
    except AssertionError as e:
        message = str(e)
        message += "\nNote that command line args are not supported while running in gradio reload mode, config.yaml must be used."
        raise AssertionError(message) from e
