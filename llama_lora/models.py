import os
import sys
import gc

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel

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


def get_new_base_model(base_model_name):
    if Global.ui_dev_mode:
        return

    device = get_device()

    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model_name,
            load_in_8bit=Global.load_8bit,
            torch_dtype=torch.float16,
            # device_map="auto",
            device_map={'': 0},  # ? https://github.com/tloen/alpaca-lora/issues/21
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model_name,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model_name, device_map={"": device}, low_cpu_mem_usage=True
        )

    model.config.pad_token_id = get_tokenizer(base_model_name).pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    return model


def get_tokenizer(base_model_name):
    if Global.ui_dev_mode:
        return

    loaded_tokenizer = Global.loaded_tokenizers.get(base_model_name)
    if loaded_tokenizer:
        return loaded_tokenizer

    tokenizer = LlamaTokenizer.from_pretrained(base_model_name)
    Global.loaded_tokenizers.set(base_model_name, tokenizer)

    return tokenizer


def get_model(
    base_model_name,
    peft_model_name = None):
    if Global.ui_dev_mode:
        return

    if peft_model_name == "None":
        peft_model_name = None

    model_key = base_model_name
    if peft_model_name:
        model_key = f"{base_model_name}//{peft_model_name}"

    loaded_model = Global.loaded_models.get(model_key)
    if loaded_model:
        return loaded_model

    peft_model_name_or_path = peft_model_name

    lora_models_directory_path = os.path.join(Global.data_dir, "lora_models")
    possible_lora_model_path = os.path.join(lora_models_directory_path, peft_model_name)
    if os.path.isdir(possible_lora_model_path):
        peft_model_name_or_path = possible_lora_model_path

    Global.loaded_models.prepare_to_set()
    clear_cache()

    model = get_new_base_model(base_model_name)

    if peft_model_name:
        device = get_device()

        if device == "cuda":
            model = PeftModel.from_pretrained(
                model,
                peft_model_name_or_path,
                torch_dtype=torch.float16,
                device_map={'': 0},  # ? https://github.com/tloen/alpaca-lora/issues/21
            )
        elif device == "mps":
            model = PeftModel.from_pretrained(
                model,
                peft_model_name_or_path,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
        else:
            model = PeftModel.from_pretrained(
                model,
                peft_model_name_or_path,
                device_map={"": device},
            )

    model.config.pad_token_id = get_tokenizer(base_model_name).pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not Global.load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    Global.loaded_models.set(model_key, model)
    clear_cache()

    return model


def clear_cache():
    gc.collect()

    # if not shared.args.cpu: # will not be running on CPUs anyway
    with torch.no_grad():
        torch.cuda.empty_cache()


def unload_models():
    Global.loaded_models.clear()
    Global.loaded_tokenizers.clear()
    clear_cache()
