import pdb
from typing import Any, Dict, Union

# import importlib
# import os
# import sys
import gc
# import re
import json
import hashlib


from transformers import (
    PreTrainedModel, AutoModelForCausalLM,
    PreTrainedTokenizerBase, AutoTokenizer, LlamaTokenizer,
)

from .config import Config
from .globals import Global
# from .lib.get_device import get_device
from .lazy_import import get_torch, get_peft
from .utils.data_processing import deep_sort_dict


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
            # use_auth_token=Config.hf_access_token
        )
    except Exception as e:
        if 'LLaMATokenizer' in str(e):
            tokenizer = LlamaTokenizer.from_pretrained(
                name_or_path,
                # use_auth_token=Config.hf_access_token
            )
        else:
            raise e

    Global.loaded_tokenizers.set(name_or_path, tokenizer)

    return tokenizer


def get_model(
    name_or_path: str,
    args: Dict[str, Any],
    adapter_model_name_or_path: Union[str, None] = None,
) -> PreTrainedModel:
    if Config.ui_dev_mode:
        raise Exception("Cannot load model in UI dev mode.")

    if Global.is_train_starting or Global.is_training:
        raise Exception("Cannot load new base model while training.")

    sorted_args = deep_sort_dict(args)
    sorted_args_json = \
        json.dumps(sorted_args, ensure_ascii=False).encode('utf-8')
    args_hash = hashlib.sha256(sorted_args_json).hexdigest()

    key = f"{name_or_path}|{args_hash}"
    model = Global.loaded_models.get(key)
    if not model:
        torch = get_torch()
        torch_dtype = args.get('torch_dtype')
        if torch_dtype and torch_dtype != 'auto':
            args['torch_dtype'] = getattr(torch, torch_dtype)
        Global.loaded_models.make_space()
        # print(f"Loading model '{name_or_path}' with args {args}...")
        model = AutoModelForCausalLM.from_pretrained(
            name_or_path,
            **args
        )
        Global.loaded_models.set(key, model)

    peft = get_peft()
    PeftModel = peft.PeftModel
    if adapter_model_name_or_path:
        adapter_name = hashlib.sha256(
            adapter_model_name_or_path.encode('utf-8')).hexdigest()
        if isinstance(model, PeftModel):
            model.base_model.enable_adapter_layers()
            if adapter_name not in model.peft_config:
                model.load_adapter(adapter_model_name_or_path,
                                   adapter_name=adapter_name)
        else:
            model = PeftModel.from_pretrained(
                model,
                adapter_model_name_or_path,
                adapter_name=adapter_name)
            Global.loaded_models.set(key, model)
        model.set_adapter(adapter_name)
    elif isinstance(model, PeftModel):
        model.base_model.disable_adapter_layers()
        model = model.base_model

    return model


def clear_loaded_models():
    Global.loaded_models.clear()
