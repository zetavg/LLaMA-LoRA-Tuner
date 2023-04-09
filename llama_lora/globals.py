import os
import subprocess

from typing import Any, Dict, List, Optional, Tuple, Union

from numba import cuda
import nvidia_smi

from .lib.finetune import train


class Global:
    version = None

    base_model: str = ""
    data_dir: str = ""
    load_8bit: bool = False

    loaded_tokenizer: Any = None
    loaded_base_model: Any = None

    # Functions
    train_fn: Any = train

    # Training Control
    should_stop_training = False

    # Generation Control
    should_stop_generating = False
    generation_force_stopped_at = None

    # Model related
    model_has_been_used = False
    loaded_base_model_with_lora = None
    loaded_base_model_with_lora_name = None

    # GPU Info
    gpu_cc = None  # GPU compute capability
    gpu_sms = None  # GPU total number of SMs
    gpu_total_cores = None  # GPU total cores
    gpu_total_memory = None

    # UI related
    ui_title: str = "LLaMA-LoRA"
    ui_emoji: str = "🦙🎛️"
    ui_subtitle: str = "Toolkit for evaluating and fine-tuning LLaMA models with low-rank adaptation (LoRA)."
    ui_show_sys_info: bool = True
    ui_dev_mode: bool = False
    ui_dev_mode_title_prefix: str = "[UI DEV MODE] "


def get_package_dir():
    current_file_path = os.path.abspath(__file__)
    parent_directory_path = os.path.dirname(current_file_path)
    return os.path.abspath(parent_directory_path)


def get_git_commit_hash():
    try:
        original_cwd = os.getcwd()
        project_dir = get_package_dir()
        try:
            os.chdir(project_dir)
            commit_hash = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
            return commit_hash
        except Exception as e:
            print(f"Cannot get git commit hash: {e}")
        finally:
            os.chdir(original_cwd)
    except Exception as e:
        print(f"Cannot get git commit hash: {e}")


commit_hash = get_git_commit_hash()

if commit_hash:
    Global.version = commit_hash[:8]


def load_gpu_info():
    try:
        cc_cores_per_SM_dict = {
            (2, 0): 32,
            (2, 1): 48,
            (3, 0): 192,
            (3, 5): 192,
            (3, 7): 192,
            (5, 0): 128,
            (5, 2): 128,
            (6, 0): 64,
            (6, 1): 128,
            (7, 0): 64,
            (7, 5): 64,
            (8, 0): 64,
            (8, 6): 128,
            (8, 9): 128,
            (9, 0): 128
        }
        # the above dictionary should result in a value of "None" if a cc match
        # is not found.  The dictionary needs to be extended as new devices become
        # available, and currently does not account for all Jetson devices
        device = cuda.get_current_device()
        device_sms = getattr(device, 'MULTIPROCESSOR_COUNT')
        device_cc = device.compute_capability
        cores_per_sm = cc_cores_per_SM_dict.get(device_cc)
        total_cores = cores_per_sm*device_sms
        print("GPU compute capability: ", device_cc)
        print("GPU total number of SMs: ", device_sms)
        print("GPU total cores: ", total_cores)
        Global.gpu_cc = device_cc
        Global.gpu_sms = device_sms
        Global.gpu_total_cores = total_cores

        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        total_memory = info.total

        total_memory_mb = total_memory / (1024 ** 2)
        total_memory_gb = total_memory / (1024 ** 3)

        # Print the memory size
        print(
            f"GPU total memory: {total_memory} bytes ({total_memory_mb:.2f} MB) ({total_memory_gb:.2f} GB)")
        Global.gpu_total_memory = total_memory

    except Exception as e:
        print(f"Notice: cannot get GPU info: {e}")


load_gpu_info()
