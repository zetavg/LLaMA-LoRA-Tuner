from typing import Any, Dict, List, Union

import os
import re
import shutil
import fnmatch
import json
import time
import jsonschema
import hashlib

from .utils.data_processing import deep_merge_dicts

from .config import Config
from .dataclasses import ModelPreset


def init_data_dir():
    os.makedirs(Config.data_dir, exist_ok=True)
    current_file_path = os.path.abspath(__file__)
    parent_directory_path = os.path.dirname(current_file_path)
    project_dir_path = os.path.abspath(
        os.path.join(parent_directory_path, "..", ".."))
    sample_data_dir_path = os.path.join(project_dir_path, "sample_data")
    copy_sample_data_if_not_exists(
        os.path.join(sample_data_dir_path, "model_presets"),
        os.path.join(Config.data_dir, "model_presets"))
    # copy_sample_data_if_not_exists(
    #     os.path.join(sample_data_dir_path, "templates"),
    #     os.path.join(Config.data_dir, "templates"))
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


def get_model_preset_settings() -> Dict[str, Any]:
    settings_file_path = os.path.join(
        Config.model_presets_path, '_settings.json')
    if not os.path.isfile(settings_file_path):
        return {}
    with open(settings_file_path, 'r') as f:
        settings = json.load(f)
    return settings


def update_model_preset_settings(new_settings):
    old_settings = get_model_preset_settings()
    settings = deep_merge_dicts(old_settings, new_settings)
    settings_file_path = os.path.join(
        Config.model_presets_path, '_settings.json')
    with open(settings_file_path, 'w') as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)


def get_model_presets() -> List[ModelPreset]:
    all_files = os.listdir(Config.model_presets_path)
    json_files = [
        name for name in all_files
        if name.endswith('.json') and not name.startswith('_')
    ]
    data = []
    for json_file in json_files:
        data.append(_get_validated_model_preset(json_file))
    sorted_data = sorted(data, key=lambda x: x.get('name'))

    model_preset_settings = get_model_preset_settings()
    default_preset_uid = model_preset_settings.get('default_preset_uid')
    starred_preset_uids = model_preset_settings.get('starred_preset_uids', [])

    model_presets = []
    default_model_presets = []
    starred_model_presets = []

    for d in sorted_data:
        model_preset = ModelPreset(d)

        is_default = model_preset.uid == default_preset_uid
        is_starred = model_preset.uid in starred_preset_uids

        model_preset.data['_is_default'] = is_default
        model_preset.data['_is_starred'] = is_starred

        if is_default:
            default_model_presets.append(model_preset)
        elif is_starred:
            starred_model_presets.append(model_preset)
        else:
            model_presets.append(model_preset)

    return default_model_presets + starred_model_presets + model_presets


def get_model_preset_choices() -> List[str]:
    return [
        f"{p.name} ({p.uid})"
        for p in get_model_presets()
    ]


def get_model_preset(uid) -> ModelPreset:
    all_files = os.listdir(Config.model_presets_path)
    json_files = [name for name in all_files if name.endswith(f'-{uid}.json')]
    if not json_files:
        raise ValueError(f"Model preset with uid \"{uid}\" not found.")
    json_file = json_files[0]
    data = _get_validated_model_preset(json_file)
    return ModelPreset(data)


def get_model_preset_from_choice(choice) -> Union[ModelPreset, None]:
    if not choice:
        return None
    match = re.search(r'\(([^()]+)\)$', choice)
    if not match:
        raise ValueError(f"Invalid model preset choice: \"{choice}\".")
    uid = match.group(1)
    return get_model_preset(uid)


def save_model_preset(uid, data, original_file_name):
    data = {k: v for k, v in data.items() if not k.startswith("_")}
    jsonschema.validate(instance=data, schema=model_preset_schema)
    file_name = data['name']
    file_name = file_name.lower()
    file_name = re.sub(r'[^a-z0-9_]+', '-', file_name)
    file_name = file_name[:40]
    file_name = file_name.strip('-_')
    file_name = f"{file_name}-{uid}.json"
    file_path = os.path.join(
        Config.model_presets_path, file_name)

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Model preset saved to \"{file_path}\".")

    if original_file_name and file_name != original_file_name:
        original_file_path = os.path.join(
            Config.model_presets_path, original_file_name
        )
        os.remove(original_file_path)
        print(f"Removed old model preset file \"{original_file_path}\".")


def delete_model_preset(uid):
    all_files = os.listdir(Config.model_presets_path)
    json_files = [name for name in all_files if name.endswith(f'-{uid}.json')]
    if not json_files:
        raise ValueError(f"Model preset with uid \"{uid}\" not found.")

    if len(json_files) > 1:
        raise ValueError(
            f"Multiple model presets matches uid \"{uid}\": {json_files}, you'll have to delete them manually.")

    json_file = json_files[0]

    json_file_path = os.path.join(
        Config.model_presets_path, json_file
    )
    os.remove(json_file_path)
    print(f"Deleted model preset file \"{json_file_path}\".")


def get_new_model_preset():
    return {
        '_uid': get_new_model_preset_uid(),
        '_file_name': None,
        'name': 'New Model Preset',
        'model': {
            'name_or_path': '',
            'args': {},
        }
    }


def get_new_model_preset_uid(existing_preset_uids=None):
    random_bytes = os.urandom(16)
    hash_object = hashlib.sha256(random_bytes)
    hex_dig = hash_object.hexdigest()
    new_uid = hex_dig[:8]

    if not existing_preset_uids:
        existing_presets = get_model_presets()
        existing_preset_uids = [preset.uid for preset in existing_presets]
    if new_uid in existing_preset_uids:
        # Generate a new uid if this one already exists.
        return get_new_model_preset_uid(
            existing_preset_uids=existing_preset_uids,
        )
    else:
        return new_uid


def get_prompt_template_names():
    all_files = os.listdir(Config.prompt_templates_path)
    names = [
        filename.rstrip(".json") for filename in all_files
        if not filename.startswith('_') and (
            fnmatch.fnmatch(filename, "*.json")
            or fnmatch.fnmatch(filename, "*.txt")
            or fnmatch.fnmatch(filename, "*.py")
        )
    ]
    return sorted(names)


def get_prompt_templates_settings() -> Dict[str, Any]:
    settings_file_path = os.path.join(
        Config.prompt_templates_path, '_settings.json')
    if not os.path.isfile(settings_file_path):
        return {}
    with open(settings_file_path, 'r') as f:
        settings = json.load(f)
    return settings


def update_prompt_templates_settings(new_settings):
    old_settings = get_prompt_templates_settings()
    settings = deep_merge_dicts(old_settings, new_settings)
    settings_file_path = os.path.join(
        Config.prompt_templates_path, '_settings.json')
    with open(settings_file_path, 'w') as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)


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


def _get_validated_model_preset(json_file_name):
    f_match = re.search(r'([^-]+)\.json$', json_file_name)
    if not f_match:
        raise ValueError(
            f"Invalid model preset file name: '{json_file_name}', name must be in format '<name>-<uid>.json'")
    uid = f_match.group(1)
    try:
        json_file_path = os.path.join(
            Config.model_presets_path, json_file_name)
        with open(json_file_path, 'r') as file:
            json_data = json.load(file)
            jsonschema.validate(instance=json_data, schema=model_preset_schema)
            return {
                **json_data,
                '_file_name': json_file_name,
                '_uid': uid,
            }
    except Exception as e:
        raise ValueError(
            f"Error reading model preset file '{json_file_path}': {str(e)}"
        ) from e


model_preset_schema = {
    'type': 'object',
    'required': [
        'name',
        'model',
    ],
    'properties': {
        'name': {'type': 'string', 'minLength': 1},
        'model': {
            'type': 'object',
            'required': [
                'name_or_path',
            ],
            'properties': {
                'name_or_path': {'type': 'string', 'minLength': 1},
                'args': {'type': 'object'},
            }
        },
    },
}
