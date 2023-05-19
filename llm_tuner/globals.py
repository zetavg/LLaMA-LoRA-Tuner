import importlib
import os
import subprocess
import psutil
import math

from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import TrainingArguments
from numba import cuda
import nvidia_smi

from .dynamic_import import dynamic_import
from .config import Config
from .utils.lru_cache import LRUCache
from .utils.eta_predictor import ETAPredictor


class Global:
    """
    A singleton class holding global states.
    """

    version: Union[str, None] = None

    base_model_name: str = ""
    tokenizer_name: Union[str, None] = None

    # Functions
    inference_generate_fn: Any
    finetune_train_fn: Any

    # Training Control
    should_stop_training: bool = False

    # Training Status
    is_train_starting: bool = False
    is_training: bool = False
    train_started_at: float = 0.0
    training_error_message: Union[str, None] = None
    training_error_detail: Union[str, None] = None
    training_total_epochs: int = 0
    training_current_epoch: float = 0.0
    training_total_steps: int = 0
    training_current_step: int = 0
    training_progress: float = 0.0
    training_log_history: List[Any] = []
    training_status_text: str = ""
    training_eta_predictor = ETAPredictor()
    training_eta: Union[int, None] = None
    training_args: Union[TrainingArguments, None] = None
    train_output: Union[None, Any] = None
    train_output_str: Union[None, str] = None
    training_params_info_text: str = ""

    # Generation Control
    should_stop_generating: bool = False
    generation_force_stopped_at: Union[float, None] = None

    # Model related
    loaded_models = LRUCache(1)
    loaded_tokenizers = LRUCache(1)
    new_base_model_that_is_ready_to_be_used = None
    name_of_new_base_model_that_is_ready_to_be_used = None

    # GPU Info
    gpu_cc = None  # GPU compute capability
    gpu_sms = None  # GPU total number of SMs
    gpu_total_cores = None  # GPU total cores
    gpu_total_memory = None


def initialize_global():
    Global.base_model_name = Config.default_base_model_name
    commit_hash = get_git_commit_hash()

    if commit_hash:
        Global.version = commit_hash[:8]

    if not Config.ui_dev_mode:
        ModelLRUCache = dynamic_import('.utils.model_lru_cache').ModelLRUCache
        Global.loaded_models = ModelLRUCache(1)
        Global.inference_generate_fn = dynamic_import('.lib.inference').generate
        Global.finetune_train_fn = dynamic_import('.lib.finetune').train
        load_gpu_info()


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


def load_gpu_info():
    # cuda = importlib.import_module('numba').cuda
    # nvidia_smi = importlib.import_module('nvidia_smi')
    print("")
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

        available_cpu_ram = psutil.virtual_memory().available
        available_cpu_ram_mb = available_cpu_ram / (1024 ** 2)
        available_cpu_ram_gb = available_cpu_ram / (1024 ** 3)
        print(
            f"CPU available memory: {available_cpu_ram} bytes ({available_cpu_ram_mb:.2f} MB) ({available_cpu_ram_gb:.2f} GB)")
        preserve_loaded_models_count = math.floor(
            (available_cpu_ram * 0.8) / total_memory) - 1
        if preserve_loaded_models_count > 1:
            ModelLRUCache = dynamic_import('.utils.model_lru_cache').ModelLRUCache
            print(
                f"Will keep {preserve_loaded_models_count} offloaded models in CPU RAM.")
            Global.loaded_models = ModelLRUCache(preserve_loaded_models_count)
            Global.loaded_tokenizers = LRUCache(preserve_loaded_models_count)

    except Exception as e:
        print(f"Notice: cannot get GPU info: {e}")

    print("")
