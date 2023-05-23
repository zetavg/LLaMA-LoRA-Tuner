from typing import List


class AlpacaPromptTemplate:
    def __init__(self) -> None:
        self.prompt_input = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
        self.prompt_no_input = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"

    def format(self, **kwargs) -> str:
        if kwargs.get('input'):
            return self.prompt_input.format(
                instruction=kwargs['instruction'],
                input=kwargs['input'],
            )

        return self.prompt_no_input.format(
            instruction=kwargs['instruction'],
        )

    @property
    def input_variables(self) -> List[str]:
        return ['instruction', 'input']
