import os
import shutil
import fnmatch
import json

from ..config import Config


def init_data_dir():
    os.makedirs(Config.data_dir, exist_ok=True)
    current_file_path = os.path.abspath(__file__)
    parent_directory_path = os.path.dirname(current_file_path)
    project_dir_path = os.path.abspath(
        os.path.join(parent_directory_path, "..", ".."))
    sample_data_dir_path = os.path.join(project_dir_path, "sample_data")
    copy_sample_data_if_not_exists(
        os.path.join(sample_data_dir_path, "templates"),
        os.path.join(Config.data_dir, "templates"))
    copy_sample_data_if_not_exists(
        os.path.join(sample_data_dir_path, "datasets"),
        os.path.join(Config.data_dir, "datasets"))
    copy_sample_data_if_not_exists(
        os.path.join(sample_data_dir_path, "lora_models"),
        os.path.join(Config.data_dir, "lora_models"))


def copy_sample_data_if_not_exists(source, destination):
    if os.path.exists(destination):
        return

    print(f"Copying sample data to \"{destination}\"")
    shutil.copytree(source, destination)


def get_available_template_names():
    templates_directory_path = os.path.join(Config.data_dir, "templates")
    all_files = os.listdir(templates_directory_path)
    names = [
        filename.rstrip(".json") for filename in all_files
        if fnmatch.fnmatch(
            filename, "*.json") or fnmatch.fnmatch(filename, "*.py")
    ]
    return sorted(names)


def get_available_dataset_names():
    datasets_directory_path = os.path.join(Config.data_dir, "datasets")
    all_files = os.listdir(datasets_directory_path)
    names = [
        filename for filename in all_files
        if fnmatch.fnmatch(filename, "*.json")
        or fnmatch.fnmatch(filename, "*.jsonl")
    ]
    return sorted(names)


def get_available_lora_model_names():
    lora_models_directory_path = os.path.join(Config.data_dir, "lora_models")
    all_items = os.listdir(lora_models_directory_path)
    names = [
        item for item in all_items
        if os.path.isdir(
            os.path.join(lora_models_directory_path, item))
    ]
    return sorted(names)


def get_path_of_available_lora_model(name):
    datasets_directory_path = os.path.join(Config.data_dir, "lora_models")
    path = os.path.join(datasets_directory_path, name)
    if os.path.isdir(path):
        return path
    return None


def get_info_of_available_lora_model(name):
    try:
        if "/" in name:
            return None
        path_of_available_lora_model = get_path_of_available_lora_model(
            name)
        if not path_of_available_lora_model:
            return None

        with open(
            os.path.join(path_of_available_lora_model, "info.json"), "r"
        ) as json_file:
            return json.load(json_file)

    except Exception as e:
        return None


def get_dataset_content(name):
    file_name = os.path.join(Config.data_dir, "datasets", name)
    if not os.path.exists(file_name):
        raise ValueError(
            f"Can't read {file_name} from datasets. File does not exist.")

    with open(file_name, "r") as file:
        if fnmatch.fnmatch(name, "*.json"):
            return json.load(file)

        elif fnmatch.fnmatch(name, "*.jsonl"):
            data = []
            for line_number, line in enumerate(file, start=1):
                try:
                    data.append(json.loads(line))
                except Exception as e:
                    raise ValueError(
                        f"Error parsing JSON on line {line_number}: {e}")
            return data
        else:
            raise ValueError(
                f"Unknown file format: {file_name}. Expects '*.json' or '*.jsonl'"
            )
