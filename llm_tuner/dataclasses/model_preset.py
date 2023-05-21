from typing import Any, Union, Dict

from ..config import Config
from ..lazy_import import get_torch


class ModelPreset:
    def __init__(self, data):
        self.data = data

    @property
    def uid(self) -> str:
        return self.data['_uid']

    @property
    def name(self) -> str:
        return self.data['name']

    @property
    def model_name_or_path(self) -> str:
        return self.data['model']['name_or_path']

    @property
    def adapter_model_name_or_path(self) -> Union[str, None]:
        use_adapter_model = self.data.get('use_adapter_model')
        if use_adapter_model is False:
            return None

        adapter_model = self.data.get('adapter_model')
        if not adapter_model:
            return None

        return self.data['adapter_model'].get('name_or_path')

    @property
    def model_args(self) -> Dict[str, Any]:
        args = self.data['model'].get('args')
        if not args:
            args = {}

        if args.get('load_in_8bit') is None:
            args['load_in_8bit'] = Config.load_8bit

        if not Config.ui_dev_mode:
            torch = get_torch()

            torch_dtype = args.get('torch_dtype')

            if torch_dtype and torch_dtype != 'auto':
                args['torch_dtype'] = getattr(torch, torch_dtype)

        return args
