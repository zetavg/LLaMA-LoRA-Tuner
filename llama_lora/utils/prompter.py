"""
A dedicated helper to manage templates and prompt building.
From https://github.com/tloen/alpaca-lora/blob/main/utils/prompter.py
"""

import json
import os.path as osp
import importlib
import itertools
from typing import Union, List, Dict

from ..config import Config
from ..globals import Global


class Prompter(object):
    __slots__ = ("template_name", "template", "template_module", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            template_name = "None"
        if template_name == "None":
            self.template_name = "None"
            return
        self.template_name = template_name
        self.template_module = None

        base_filename, ext = osp.splitext(template_name)
        if ext == "":
            filename = base_filename + ".json"
        else:
            filename = base_filename + ext

        file_path = osp.join(Config.data_dir, "templates", filename)

        if not osp.exists(file_path):
            raise ValueError(f"Can't read {file_path}")

        if ext == ".py":
            importlib_util = importlib.util  # type: ignore
            template_module_spec = importlib_util.spec_from_file_location(
                "template_module", file_path)
            template_module = importlib_util.module_from_spec(
                template_module_spec)
            template_module_spec.loader.exec_module(template_module)
            self.template_module = template_module

            if not hasattr(template_module, "variables"):
                raise ValueError(
                    "The template module does not have a \"variables\" attribute.")

            self.template = {
                'variables': template_module.variables
            }

            if hasattr(template_module, "response_split"):
                self.template["response_split"] = template_module.response_split

            return

        with open(file_path) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        variables: Union[Dict[str, str], List[Union[None, str]]] = [],
        # instruction: str,
        # input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        if self.template_name == "None":
            if type(variables) == list:
                res = get_val(variables, 0, "")
            elif type(variables) == dict:
                res = variables.get("prompt", "")
            else:
                raise ValueError(f"Invalid variables type: {type(variables)}")
        elif "variables" in self.template:
            variable_names = self.template.get("variables")
            # if type(variable_names) != list:
            #     raise ValueError(f"Invalid variable_names type {type(variable_names)} defined in template {self.template_name}, expecting list.")
            if self.template_module:
                if type(variables) == list:
                    variables = {k: v for k, v in zip(
                        variable_names, variables)}

                res = self.template_module.get_prompt(variables)
            else:
                if type(variables) == dict:
                    variables = [variables.get(name, None)
                                 for name in variable_names]

                if "default" not in self.template:
                    raise ValueError(
                        f"The template {self.template_name} has \"variables\" defined but does not has a default prompt defined. Please do it like: '\"default\": \"prompt_with_instruction\"' to handle cases when a matching prompt can't be found.")
                default_prompt_name = self.template.get("default")
                if default_prompt_name not in self.template:
                    raise ValueError(
                        f"The template {self.template_name} has \"default\" set to \"{default_prompt_name}\" but it's not defined. Please do it like: '\"{default_prompt_name}\": \"...\".")
                prompt_name = get_prompt_name(variables, variable_names)
                prompt_template = self.template.get(default_prompt_name)
                if prompt_name in self.template:
                    prompt_template = self.template.get(prompt_name)

                res = prompt_template.format(
                    **variables_to_dict(variables, variable_names))

        else:
            if type(variables) == dict:
                instruction = variables.get("instruction", "")
                input = variables.get("input")
            else:
                instruction = get_val(variables, 0, "")
                input = get_val(variables, 1)
            # returns the full prompt from instruction and optional input
            # if a label (=response, =output) is provided, it's also appended.
            if input:
                res = self.template["prompt_input"].format(
                    instruction=instruction, input=input
                )
            else:
                res = self.template["prompt_no_input"].format(
                    instruction=instruction
                )

        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        if self.template_name == "None":
            return output

        splitted_output = output.split(self.template["response_split"])
        # if len(splitted_output) <= 1:
        #     return output.strip()

        return self.template["response_split"].join(
            splitted_output[1:]
        ).strip()

    def get_variable_names(self) -> List[str]:
        if self.template_name == "None":
            return ["prompt"]
        elif "variables" in self.template:
            return self.template['variables']
        else:
            return ["instruction", "input"]

    def get_train_data_from_dataset(self, data, only_first_n_items=None):
        if self.template_module:
            if hasattr(self.template_module,
                       "get_train_data_list_from_dataset"):
                data = self.template_module.get_train_data_list_from_dataset(
                    data)
            if only_first_n_items:
                data = data[:only_first_n_items]
            return list(itertools.chain(*list(
                map(self.template_module.get_train_data, data)
            )))

        if only_first_n_items:
            data = data[:only_first_n_items]

        data = process_json_dataset(data)

        train_data = [
            {
                'prompt': self.generate_prompt(d['variables']),
                'completion': d['output'],
                **{"_var_" + k: v for k, v in d['variables'].items()}
            }
            for d in data]

        return train_data


def get_val(arr, index, default=None):
    return arr[index] if -len(arr) <= index < len(arr) else default


def get_prompt_name(variables, variable_names):
    result = [y for x, y in zip(
        variables, variable_names) if x not in (None, '')]
    return "prompt_with_" + '_'.join(result)


def variables_to_dict(variables, variable_names):
    return {
        key: (variables[i] if i < len(variables)
              and variables[i] is not None else '')
        for i, key in enumerate(variable_names)
    }


def process_json_dataset(data):
    if not isinstance(data, list):
        raise ValueError("The dataset is not an array of objects.")

    first_item = get_val_from_arr(data, 0, None)

    if first_item is None:
        raise ValueError("The dataset is empty.")
    if not isinstance(first_item, dict):
        raise ValueError("The dataset is not an array of objects.")

    # Convert OpenAI fine-tuning dataset to LLaMA LoRA style
    if "completion" in first_item and "output" not in first_item:
        data = [
            {"output" if k == "completion" else k: v for k, v in d.items()}
            for d in data]
        first_item = get_val_from_arr(data, 0, None)

    # Flatten Stanford Alpaca style instances
    if "instances" in first_item and isinstance(first_item["instances"], list):
        data = [
            {"output" if k == "completion" else k: v for k, v in d.items()}
            for d in data]
        flattened_data = []
        for item in data:
            for instance in item["instances"]:
                d = {k: v for k, v in item.items() if k != "instances"}
                d.update(instance)
                flattened_data.append(d)
        data = flattened_data
        first_item = get_val_from_arr(data, 0, None)

    if "output" not in first_item:
        raise ValueError(
            "The data does not contains an \"output\" or \"completion\".")

    # Put all variables under the "variables" key if it does not exists
    if "variables" not in first_item:
        data = [
            {
                "variables":
                    {k: v for k, v in d.items() if k != "output"},
                "output":
                    d["output"]
            }
            for d in data
        ]
    return data


def get_val_from_arr(arr, index, default=None):
    return arr[index] if -len(arr) <= index < len(arr) else default
