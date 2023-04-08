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

    if Global.loaded_base_model_with_lora and Global.loaded_base_model_with_lora_name == lora_weights:
        return Global.loaded_base_model_with_lora

    if device == "cuda":
        model = PeftModel.from_pretrained(
            get_base_model(),
            lora_weights,
            torch_dtype=torch.float16,
            device_map={'': 0},  # ? https://github.com/tloen/alpaca-lora/issues/21
        )
    elif device == "mps":
        model = PeftModel.from_pretrained(
            get_base_model(),
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = PeftModel.from_pretrained(
            get_base_model(),
            lora_weights,
            device_map={"": device},
        )

    model.config.pad_token_id = get_tokenizer().pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not Global.load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    Global.loaded_base_model_with_lora = model
    Global.loaded_base_model_with_lora_name = lora_weights
    return model


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
                Global.base_model, device_map={"": device}, low_cpu_mem_usage=True
            )

        Global.loaded_base_model.config.pad_token_id = get_tokenizer().pad_token_id = 0
        Global.loaded_base_model.config.bos_token_id = 1
        Global.loaded_base_model.config.eos_token_id = 2


def clear_cache():
    gc.collect()

    # if not shared.args.cpu: # will not be running on CPUs anyway
    with torch.no_grad():
        torch.cuda.empty_cache()


def unload_models():
    del Global.loaded_base_model
    Global.loaded_base_model = None

    del Global.loaded_tokenizer
    Global.loaded_tokenizer = None

    del Global.loaded_base_model_with_lora
    Global.loaded_base_model_with_lora = None

    Global.loaded_base_model_with_lora_name = None

    clear_cache()

    Global.model_has_been_used = False


def unload_models_if_already_used():
    if Global.model_has_been_used:
        unload_models()
