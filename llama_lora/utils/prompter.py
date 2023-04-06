"""
A dedicated helper to manage templates and prompt building.
From https://github.com/tloen/alpaca-lora/blob/main/utils/prompter.py
"""

import json
import os.path as osp
from typing import Union, List

from ..globals import Global


class Prompter(object):
    __slots__ = ("template_name", "template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            template_name = "None"
        if template_name == "None":
            self.template_name = "None"
            return
        self.template_name = template_name

        file_name = osp.join(Global.data_dir, "templates",
                             f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        variables: List[Union[None, str]] = [],
        # instruction: str,
        # input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        if self.template_name == "None":
            if type(variables) == list:
                res = get_val(variables, 0, "")
            else:
                res = variables.get("prompt", "")
        elif "variables" in self.template:
            variable_names = self.template.get("variables")
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
        return self.template["response_split"].join(
            output.split(self.template["response_split"])[1:]
        ).strip()

    def get_variable_names(self) -> List[str]:
        if self.template_name == "None":
            return ["prompt"]
        elif "variables" in self.template:
            return self.template.get("variables")
        else:
            return ["instruction", "input"]


def get_val(arr, index, default=None):
    return arr[index] if -len(arr) <= index < len(arr) else default


def get_prompt_name(variables, variable_names):
    result = [y for x, y in zip(
        variables, variable_names) if x not in (None, '')]
    return "prompt_with_" + '_'.join(result)


def variables_to_dict(variables, variable_names):
    return {key: (variables[i] if i < len(variables) and variables[i] is not None else '') for i, key in enumerate(variable_names)}
