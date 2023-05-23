import pdb
from typing import Union, List, Dict

import os
import inspect
import json
import json5
import importlib
import hashlib
import itertools

from langchain.prompts import PromptTemplate
# from numba.core import base

from ..config import Config


class Prompter:
    def __init__(self, prompt_template_name: str):
        self.prompt_template_name = prompt_template_name
        self.prompt_template_filename = None
        self.prompt_template_type = None
        self.prompt_template = None

        if prompt_template_name == 'None':
            self.prompt_template_type = 'None'
            return

        base_filename, ext = os.path.splitext(prompt_template_name)
        if not ext:
            ext = ".json"

        filename = base_filename + ext
        self.prompt_template_filename = filename

        filepath = os.path.join(Config.prompt_templates_path, filename)

        if not os.path.exists(filepath):
            raise ValueError(f"Prompt template '{filepath}' does not exists.")

        if ext == '.py':
            module_spec = importlib.util.spec_from_file_location(
                # hashlib.sha256(filepath.encode('utf-8')).hexdigest(),
                sha1_hash_of_file(filepath),
                filepath,
            )
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            classes = [
                attr
                for name, attr in inspect.getmembers(module)
                if inspect.isclass(attr)]

            if not classes:
                raise ValueError(
                    f"No class is defined in the prompt template file '{filepath}'.")

            if len(classes) > 1:
                print(
                    f"Mutiple classes is defined in the prompt template file '{filepath}'. Using the first one which is {classes[0].__name__}.")

            klass = classes[0]

            self.prompt_template_type = 'prompt'
            self.prompt_template = klass()

        elif ext == '.json':
            with open(filepath, 'r') as f:
                json = json5.load(f)
            t = json.get('_type')
            self.prompt_template_type = t
            if not t:
                raise ValueError(
                    f"Invalid prompt template '{filepath}': missing \"_type\" attribute.")
            if t == 'prompt':
                self.prompt_template = PromptTemplate(**{
                    k: v for k, v in json.items() if k != '_type'
                })
            else:
                raise ValueError(
                    f"Unknown prompt template type: '{t}' ('{filepath}').")

        elif ext == '.txt':
            with open(filepath, 'r') as f:
                file_contents = f.read()
            self.prompt_template_type = 'prompt'
            self.prompt_template = PromptTemplate.from_template(file_contents)

        else:
            raise ValueError(
                f"Unknown prompt template file extension: {ext} ('{filepath}'')")

        # assert self.prompt_template, 'Prompt template is not set by __init__.'

    def get_variable_names(self) -> List[str]:
        if self.prompt_template_name == 'None':
            return ['prompt']
        elif 'prompt':
            input_variables = self.prompt_template.input_variables
            if not is_list_of_strings(input_variables):
                raise ValueError(
                    f"Expect {self.prompt_template.__class__.__name__}.input_variables to be a list of strings, but got {input_variables}.")

            return input_variables

    def generate_prompt(
        self,
        variables: Union[Dict[str, str], List[Union[None, str]]] = [],
    ) -> str:
        if self.prompt_template_type == 'None':
            if not isinstance(variables, list):
                variables = variables.values()
            return ''.join(variables)
        elif self.prompt_template_type == 'prompt':
            if not isinstance(variables, dict):
                variable_names = self.get_variable_names()
                variables = {
                    k: v
                    for k, v in
                    itertools.zip_longest(
                        variable_names,
                        variables[:len(variable_names)],
                        fillvalue=''
                    )
                }
            prompt = self.prompt_template.format(**variables)
            return prompt
        else:
            raise NotImplementedError('')

    def get_response(
        self,
        output: str,
        input_variables: Union[Dict[str, str], List[Union[None, str]]] = [],
    ):
        if self.prompt_template_type == 'None':
            return output

        origional_prompt = self.generate_prompt(input_variables)
        return remove_common_from_start(origional_prompt, output)

    # @property
    # def samples(self) -> List[List[str]]:
    #     prompt_templates_settings = get_prompt_templates_settings()
    #     all_samples = None
    #     if prompt_templates_settings:
    #         all_samples = prompt_templates_settings.get('samples')
    #     samples = []
    #     if all_samples:
    #         samples = (
    #             all_samples.get(self.prompt_template_name, [])
    #             or all_samples.get(self.prompt_template_filename, [])
    #         )
    #         if not isinstance(samples, list):
    #             print(
    #                 f"WARNING: samples for prompt template '{self.prompt_template_name}' is not a list. Ignoring.")
    #             return []
    #         variable_names = self.get_variable_names()
    #         samples = [[
    #             n
    #             for _, n in
    #             itertools.zip_longest(
    #                 variable_names,
    #                 s[:len(variable_names)] if isinstance(s, list) else [s],
    #                 fillvalue=''
    #             )
    #         ] for s in samples]
    #     return samples


def remove_common_from_start(str1, str2):
    i = 0
    while i < len(str1) and i < len(str2) and str1[i] == str2[i]:
        i += 1
    return str2[i:]


def is_list_of_strings(lst):
    return isinstance(lst, list) and all(isinstance(item, str) for item in lst)


def sha1_hash_of_file(file_path):
    hash_sha1 = hashlib.sha1()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha1.update(chunk)
    return hash_sha1.hexdigest()
