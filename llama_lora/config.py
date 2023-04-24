import os
import pytz
from typing import List, Union, Any


class Config:
    """
    Stores the application configuration. This is a singleton class.
    """

    data_dir: str = ""
    load_8bit: bool = False

    default_base_model_name: str = ""
    base_model_choices: Union[List[str], str] = []

    trust_remote_code: bool = False

    timezone: Any = pytz.UTC

    auth_username: Union[str, None] = None
    auth_password: Union[str, None] = None

    # WandB
    enable_wandb: Union[bool, None] = None
    wandb_api_key: Union[str, None] = None
    default_wandb_project: str = "llama-lora-tuner"

    # UI related
    ui_title: str = "LLaMA-LoRA Tuner"
    ui_emoji: str = "🦙🎛️"
    ui_subtitle: str = "Toolkit for evaluating and fine-tuning LLaMA models with low-rank adaptation (LoRA)."
    ui_show_sys_info: bool = True
    ui_dev_mode: bool = False
    ui_dev_mode_title_prefix: str = "[UI DEV MODE] "


def process_config():
    Config.data_dir = os.path.abspath(Config.data_dir)

    if isinstance(Config.base_model_choices, str):
        base_model_choices = Config.base_model_choices.split(',')
        base_model_choices = [name.strip() for name in base_model_choices]
        Config.base_model_choices = base_model_choices

    if isinstance(Config.timezone, str):
        Config.timezone = pytz.timezone(Config.timezone)

    if Config.default_base_model_name not in Config.base_model_choices:
        Config.base_model_choices = [
            Config.default_base_model_name] + Config.base_model_choices

    if Config.enable_wandb is None:
        if (
            Config.wandb_api_key and len(Config.wandb_api_key) > 0
                and Config.default_wandb_project and len(Config.default_wandb_project) > 0
        ):
            Config.enable_wandb = True
