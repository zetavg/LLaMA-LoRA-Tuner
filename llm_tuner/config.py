from typing import Dict, Union, Any

import os
import pytz


class ClassProperty:
    def __init__(self, getter):
        self.getter = getter

    def __get__(self, instance, owner):
        return self.getter(owner)


class Config:
    """
    Stores the application configuration. This is a singleton class.
    """

    server_name: str = "127.0.0.1"

    # Where data is stored
    data_dir: str = "./data"

    # Application Settings
    timezone: Any = pytz.UTC

    # Model Related
    default_load_in_8bit: bool = False
    default_torch_dtype: str = 'float16'

    default_generation_config = {
        'temperature': 0,
        'top_p': 0.75,
        'top_k': 40,
        'num_beams': 2,
        'repetition_penalty': 1.2,
        'max_new_tokens': 128,
    }

    # Authentication
    auth_username: Union[str, None] = None
    auth_password: Union[str, None] = None

    # Hugging Face
    # hf_access_token: Union[str, None] = None

    # WandB
    enable_wandb: Union[bool, None] = None
    wandb_api_key: Union[str, None] = None
    default_wandb_project: str = "llama-lora-tuner"

    # UI related
    ui_title: str = "LLM Tuner"
    ui_emoji: str = "🦙🎛️"
    ui_subtitle: str = \
        "Toolkit for evaluating and fine-tuning language models."
    ui_show_sys_info: bool = True
    ui_dev_mode: bool = False
    ui_dev_mode_title_prefix: str = "[UI DEV MODE] "

    @ClassProperty
    def model_presets_path(self) -> str:
        return os.path.join(self.data_dir, 'model_presets')

    @ClassProperty
    def models_path(self) -> str:
        return os.path.join(self.data_dir, 'models')

    @ClassProperty
    def adapter_models_path(self) -> str:
        return os.path.join(self.data_dir, 'adapter_models')


def set_config(config_dict: Dict[str, Any]):
    for key, value in config_dict.items():
        if key == 'default_generation_config':
            Config.default_generation_config.update(value)
            continue
        if not hasattr(Config, key):
            available_keys = [k for k in vars(
                Config) if not k.startswith('__')]
            raise ValueError(
                f"Invalid config key '{key}' in config. Available keys: {', '.join(available_keys)}.")
        setattr(Config, key, value)


def process_config():
    Config.data_dir = os.path.abspath(Config.data_dir)

    if isinstance(Config.timezone, str):
        Config.timezone = pytz.timezone(Config.timezone)

    if Config.enable_wandb is None:
        if (
            Config.wandb_api_key and len(Config.wandb_api_key) > 0
                and Config.default_wandb_project and len(Config.default_wandb_project) > 0
        ):
            Config.enable_wandb = True
