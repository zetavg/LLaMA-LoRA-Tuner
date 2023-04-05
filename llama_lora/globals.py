import os
import subprocess

from typing import Any, Dict, List, Optional, Tuple, Union

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

    # Model related
    model_has_been_used = False

    # UI related
    ui_title: str = "LLaMA-LoRA"
    ui_emoji: str = "🦙🎛️"
    ui_subtitle: str = "Toolkit for evaluating and fine-tuning LLaMA models with low-rank adaptation (LoRA)."
    ui_show_sys_info: bool = True
    ui_dev_mode: bool = False


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
