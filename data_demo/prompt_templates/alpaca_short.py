import os
import importlib.util

alpaca_module_spec = importlib.util.spec_from_file_location(
    'alpaca',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'alpaca.py'),
)
alpaca_module = importlib.util.module_from_spec(alpaca_module_spec)
alpaca_module_spec.loader.exec_module(alpaca_module)


class AlpacaShortPromptTemplate(alpaca_module.AlpacaPromptTemplate):
    def __init__(self) -> None:
        self.prompt_input = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
        self.prompt_no_input = "### Instruction:\n{instruction}\n\n### Response:\n"
