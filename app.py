from typing import Union

import os
import fire
import yaml
import gradio as gr

from llm_tuner.config import Config, set_config
from llm_tuner.initialization import initialize
from llm_tuner.ui import (
    main_page, get_page_title, get_css_styles
)


def main(
    data_dir: Union[str, None] = None,
    server_name: Union[str, None] = None,
    share: bool = False,
    skip_loading_default_model: bool = False,
    auth: Union[str, None] = None,
    default_load_in_8bit: Union[bool, None] = None,
    default_torch_dtype: Union[str, None] = None,
    ui_show_sys_info: Union[bool, None] = None,
    ui_dev_mode: Union[bool, None] = None,
    wandb_api_key: Union[str, None] = None,
    wandb_project: Union[str, None] = None,
    # hf_access_token: Union[str, None] = None,
    timezone: Union[str, None] = None,
    config: Union[str, None] = None,
):
    '''
    Start the LLM Tuner UI.

    :param data_dir: The path to the directory to store data.

    :param server_name: Allows to listen on all interfaces by providing '0.0.0.0'.
    :param share: Create a public Gradio URL.

    :param wandb_api_key: The API key for Weights & Biases. Setting either this or `wandb_project` will enable Weights & Biases.
    :param wandb_project: The default project name for Weights & Biases. Setting either this or `wandb_api_key` will enable Weights & Biases.

    :param hf_access_token: Provide an access token to load private models form Hugging Face Hub. An access token can be created at https://huggingface.co/settings/tokens.
    '''

    config_path = config
    if not config_path:
        app_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(app_dir, 'config.yaml')
    config_from_file = read_yaml_config(config_path)
    if config_from_file:
        try:
            set_config(config_from_file)
        except ValueError as e:
            raise ValueError(f"{str(e)} Check {config_path}.") from e

    if server_name is not None:
        Config.server_name = server_name

    if data_dir is not None:
        Config.data_dir = data_dir

    if default_load_in_8bit is not None:
        Config.default_load_in_8bit = default_load_in_8bit

    if default_torch_dtype is not None:
        Config.default_torch_dtype = default_torch_dtype

    if auth is not None:
        try:
            [Config.auth_username, Config.auth_password] = auth.split(':')
        except ValueError:
            raise ValueError("--auth must be in the format <username>:<password>, e.g.: --auth='username:password'")

    # if hf_access_token is not None:
    #     Config.hf_access_token = hf_access_token

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

    initialize(skip_loading_default_model=skip_loading_default_model)

    with gr.Blocks(title=get_page_title(), css=get_css_styles()) as demo:
        main_page()

    demo.queue(concurrency_count=1).launch(
        server_name=Config.server_name,
        share=share,
        auth=((Config.auth_username, Config.auth_password)
              if Config.auth_username and Config.auth_password else None)
    )


def read_yaml_config(config_path: str):
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
