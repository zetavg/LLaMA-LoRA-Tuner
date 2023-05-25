from typing import Dict, List, Union, Any

import os
import pytz

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
default_data_dir = os.path.join(project_dir, "data")
default_ui_features = [
    'inference',
    'chat',
    'models',
    'finetuning',
    'tools',
]
demo_mode_ui_features = [
    'inference',
    'chat',
    'tools',
]


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
    data_dir: str = default_data_dir

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
    default_generation_stop_sequence = None

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
    ui_features: List[str] = default_ui_features
    ui_show_starter_tooltips: bool = True
    ui_title: str = "LLM Tuner"
    ui_emoji: str = "🦙🎛️"
    ui_subtitle: str = \
        "Toolkit for evaluating and fine-tuning language models."
    ui_show_sys_info: bool = True

    # More UI options
    ui_inference_open_options_by_default: bool = True
    ui_chat_reminder_message: str = \
        'Language models may produce inaccurate information about people, places, or facts.'

    # UI dev mode
    ui_dev_mode: bool = False
    ui_dev_mode_title_prefix: str = "[UI DEV MODE] "

    # Special modes
    demo_mode: bool = False

    @ClassProperty
    def model_presets_path(self) -> str:
        return os.path.join(self.data_dir, 'model_presets')

    @ClassProperty
    def models_path(self) -> str:
        return os.path.join(self.data_dir, 'models')

    @ClassProperty
    def adapter_models_path(self) -> str:
        return os.path.join(self.data_dir, 'adapter_models')

    @ClassProperty
    def prompt_templates_path(self) -> str:
        return os.path.join(self.data_dir, 'prompt_templates')

    @ClassProperty
    def prompt_samples_path(self) -> str:
        return os.path.join(self.data_dir, 'prompt_samples')


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
    if Config.data_dir == default_data_dir:
        demo_data_dir = os.path.join(project_dir, "data_demo")
        if os.path.isdir(demo_data_dir):
            print(f"Demo Data Dir '{demo_data_dir}' detected! Using it as 'data_dir'.")
            Config.data_dir = demo_data_dir

    if Config.demo_mode:
        if Config.ui_features == default_ui_features:
            Config.ui_features = demo_mode_ui_features
        Config.ui_show_starter_tooltips = False

    Config.data_dir = os.path.abspath(Config.data_dir)

    if isinstance(Config.timezone, str):
        Config.timezone = pytz.timezone(Config.timezone)

    if (
        Config.default_generation_config
        and Config.default_generation_config.get('temperature')
        and not Config.default_generation_config.get('do_sample')
    ):
        Config.default_generation_config['do_sample'] = True

    if Config.enable_wandb is None:
        if (
            Config.wandb_api_key and len(Config.wandb_api_key) > 0
                and Config.default_wandb_project and len(Config.default_wandb_project) > 0
        ):
            Config.enable_wandb = True
