from typing import Any, Dict, List, Optional, Tuple, Union


class Global:
    base_model: str = ""
    data_dir: str = ""
    load_8bit: bool = False

    loaded_tokenizer: Any = None
    loaded_base_model: Any = None

    # UI related
    ui_title: str = "LLaMA-LoRA"
    ui_emoji: str = "🦙🎛️"
    ui_subtitle: str = "Toolkit for examining and fine-tuning LLaMA models using low-rank adaptation (LoRA)."
    ui_show_sys_info: bool = True
    ui_dev_mode: bool = False
