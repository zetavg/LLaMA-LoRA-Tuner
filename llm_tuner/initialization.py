from .config import Config, process_config
from .globals import initialize_global
from .data import init_data_dir, get_model_presets
from .utils.download_models import download_models


def initialize(skip_loading_default_model=False):
    process_config()
    initialize_global()
    assert (Config.data_dir), "data_dir is not set!"

    init_data_dir()

    if Config.ui_dev_mode:
        print("In UI dev mode.")
        print()
        return

    if Config.demo_mode and not Config.ui_dev_mode:
        # In demo mode, make sure all used models are pre-downloaded.
        download_models()

    if not skip_loading_default_model:
        load_default_model()


def load_default_model():
    model_presets = get_model_presets()
    if not model_presets:
        print("No model presets found! Will not load default model.")
        print()
        return

    default_model_preset = model_presets[0]

    print(
        f"Loading default model '{default_model_preset.model_name_or_path}'..."
    )
    default_model_preset.tokenizer
    default_model_preset.model
    print(
        f"Default model '{default_model_preset.model_name_or_path}' loaded."
    )
    print()
