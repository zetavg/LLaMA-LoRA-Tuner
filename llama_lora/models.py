import os
import sys
import gc

import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from .globals import Global


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

    try:
        if torch.backends.mps.is_available():
            return "mps"
    except:  # noqa: E722
        pass


device = get_device()


def get_base_model():
    load_base_model()
    return Global.loaded_base_model


def get_model_with_lora(lora_weights: str = "tloen/alpaca-lora-7b"):
    Global.model_has_been_used = True

    if device == "cuda":
        return PeftModel.from_pretrained(
            get_base_model(),
            lora_weights,
            torch_dtype=torch.float16,
            device_map={'': 0},  # ? https://github.com/tloen/alpaca-lora/issues/21
        )
    elif device == "mps":
        return PeftModel.from_pretrained(
            get_base_model(),
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        return PeftModel.from_pretrained(
            get_base_model(),
            lora_weights,
            device_map={"": device},
        )


def get_tokenizer():
    load_base_model()
    return Global.loaded_tokenizer


def load_base_model():
    if Global.ui_dev_mode:
        return

    if Global.loaded_tokenizer is None:
        Global.loaded_tokenizer = LlamaTokenizer.from_pretrained(
            Global.base_model
        )
    if Global.loaded_base_model is None:
        if device == "cuda":
            Global.loaded_base_model = LlamaForCausalLM.from_pretrained(
                Global.base_model,
                load_in_8bit=Global.load_8bit,
                torch_dtype=torch.float16,
                # device_map="auto",
                device_map={'': 0},  # ? https://github.com/tloen/alpaca-lora/issues/21
            )
        elif device == "mps":
            Global.loaded_base_model = LlamaForCausalLM.from_pretrained(
                Global.base_model,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
        else:
            Global.loaded_base_model = LlamaForCausalLM.from_pretrained(
                base_model, device_map={"": device}, low_cpu_mem_usage=True
            )

    # unwind broken decapoda-research config
    Global.loaded_base_model.config.pad_token_id = Global.loaded_tokenizer.pad_token_id = 0  # unk
    Global.loaded_base_model.config.bos_token_id = 1
    Global.loaded_base_model.config.eos_token_id = 2


def unload_models():
    del Global.loaded_base_model
    Global.loaded_base_model = None

    del Global.loaded_tokenizer
    Global.loaded_tokenizer = None

    gc.collect()

    # if not shared.args.cpu: # will not be running on CPUs anyway
    with torch.no_grad():
        torch.cuda.empty_cache()

    Global.model_has_been_used = False


def unload_models_if_already_used():
    if Global.model_has_been_used:
        unload_models()
