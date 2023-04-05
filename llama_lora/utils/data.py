import os
import shutil
import fnmatch
import json

from ..globals import Global


def init_data_dir():
    current_file_path = os.path.abspath(__file__)
    parent_directory_path = os.path.dirname(current_file_path)
    project_dir_path = os.path.abspath(
        os.path.join(parent_directory_path, "..", ".."))
    os.makedirs(os.path.join(Global.data_dir, "lora_models"), exist_ok=True)
    copy_sample_data_if_not_exists(os.path.join(project_dir_path, "templates"),
                                   os.path.join(Global.data_dir, "templates"))
    copy_sample_data_if_not_exists(os.path.join(project_dir_path, "datasets"),
                                   os.path.join(Global.data_dir, "datasets"))


def copy_sample_data_if_not_exists(source, destination):
    if os.path.exists(destination):
        return

    print(f"Copying sample data to \"{destination}\"")
    shutil.copytree(source, destination)


def get_available_template_names():
    templates_directory_path = os.path.join(Global.data_dir, "templates")
    all_files = os.listdir(templates_directory_path)
    return [os.path.splitext(filename)[0] for filename in all_files if fnmatch.fnmatch(filename, "*.json")]


def get_available_dataset_names():
    datasets_directory_path = os.path.join(Global.data_dir, "datasets")
    all_files = os.listdir(datasets_directory_path)
    return [filename for filename in all_files if fnmatch.fnmatch(filename, "*.json") or fnmatch.fnmatch(filename, "*.jsonl")]


def get_dataset_content(name):
    file_name = os.path.join(Global.data_dir, "datasets", name)
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
                f"Unknown file format: {file_name}. Expects '*.json' or '*.jsonl'")
