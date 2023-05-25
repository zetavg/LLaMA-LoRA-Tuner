from typing import Any

import json5


def prepare_generation_config_args(generation_config_args: Any):

    if isinstance(generation_config_args, str):
        generation_config_args = json5.loads(generation_config_args)

    if 'temperature' in generation_config_args:
        generation_config_args['temperature'] = \
            float(generation_config_args['temperature'] or 0)
        # Now controlled via JS
        # if generation_config_args['temperature'] > 0:
        #     generation_config_args['do_sample'] = True

    if 'repetition_penalty' in generation_config_args:
        generation_config_args['repetition_penalty'] = \
            float(generation_config_args['repetition_penalty'] or 0)

    return generation_config_args
