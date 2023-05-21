# import importlib
# import os
# import sys
# import gc
# import json
# import re

from transformers import (
    AutoModelForCausalLM, AutoModel,
    PreTrainedTokenizerBase, AutoTokenizer, LlamaTokenizer,
)

from .config import Config
from .globals import Global
# from .lib.get_device import get_device
# from .lazy_import import get_torch


def get_tokenizer(name_or_path: str) -> PreTrainedTokenizerBase:
    # if Config.ui_dev_mode:
    #     raise Exception("Cannot use tokenizer in UI dev mode.")

    # if Global.is_train_starting or Global.is_training:
    #     raise Exception("Cannot load tokenizer while training.")

    cache_key = name_or_path

    loaded_tokenizer = Global.loaded_tokenizers.get(cache_key)
    if loaded_tokenizer:
        return loaded_tokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            name_or_path,
            trust_remote_code=Config.trust_remote_code,
            use_auth_token=Config.hf_access_token
        )
    except Exception as e:
        if 'LLaMATokenizer' in str(e):
            tokenizer = LlamaTokenizer.from_pretrained(
                name_or_path,
                trust_remote_code=Config.trust_remote_code,
                use_auth_token=Config.hf_access_token
            )
        else:
            raise e

    Global.loaded_tokenizers.set(name_or_path, tokenizer)

    return tokenizer
