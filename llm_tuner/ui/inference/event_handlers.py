import time
import json
import gradio as gr
from transformers import GenerationConfig

from ...config import Config
from ...globals import Global
from ...data import (
    get_model_preset_choices,
    get_prompt_template_names,
    get_available_lora_model_names,
    get_info_of_available_lora_model,
    get_model_preset_from_choice,
)

from ...utils.prompter import Prompter
from ...utils.data_processing import comparing_lists
from ...utils.prepare_generation_config_args import prepare_generation_config_args


def handle_reload_selections(
    current_model_preset_selection, current_prompt_template
):
    prompt_template_names = get_prompt_template_names()
    prompt_template_choices = ["None"] + prompt_template_names

    if current_prompt_template not in prompt_template_choices:
        current_prompt_template = None

    if not current_prompt_template:
        current_prompt_template = \
            next(iter(prompt_template_choices), None)

    model_preset_choices = get_model_preset_choices()

    if current_model_preset_selection not in model_preset_choices:
        current_model_preset_selection = None

    if not current_model_preset_selection:
        current_model_preset_selection = \
            next(iter(model_preset_choices), None)

    return (
        gr.Dropdown.update(
            choices=model_preset_choices,
            value=current_model_preset_selection),
        gr.Dropdown.update(
            choices=prompt_template_choices,
            value=current_prompt_template)
    )


def handle_prompt_template_change(prompt_template, model_preset_selection):
    try:
        prompter = Prompter(prompt_template)
        var_names = prompter.get_variable_names()
        human_var_names = [
            ' '.join(
                word.capitalize()
                for word in item.split('_')
            )
            for item in var_names]
        variable_textbox_updates = [
            gr.Textbox.update(label=name, visible=True)
            for name in human_var_names]

        while len(variable_textbox_updates) < 8:
            variable_textbox_updates.append(
                gr.Textbox.update(label="Not Used", visible=False)
            )

        message_update = \
            gr.Markdown.update('', visible=False)

        model_preset = get_model_preset_from_choice(model_preset_selection)
        if (
            model_preset
            and model_preset.default_prompt_template != 'None'
            and model_preset.default_prompt_template != prompt_template
        ):
            message_update = gr.Markdown.update(
                f'<span style="font-size: 90%;">ⓘ</span> The default prompt template of the selected model is <code>{model_preset.default_prompt_template}</code>.',
                visible=True
            )

        return [message_update] + variable_textbox_updates
    except Exception as e:
        raise gr.Error(str(e)) from e


def prepare_generate(
    model_preset_choice,
    progress=gr.Progress(track_tqdm=True)
):
    try:
        model_preset = get_model_preset_from_choice(model_preset_choice)
        if model_preset:
            model_preset.tokenizer
            model_preset.model
        return ("", "", gr.Textbox.update(visible=False))

    except Exception as e:
        raise gr.Error(str(e))


def handle_generate(
    model_preset_choice,
    prompt_template,
    generation_config,
    stop_sequences,
    stream_output,
    *variables
):
    model_preset = get_model_preset_from_choice(model_preset_choice)
    if not model_preset:
        return

    if not isinstance(stop_sequences, list):
        stop_sequences = [stop_sequences]
    stop_sequences = [s for s in stop_sequences if s]

    try:
        if Global.generation_force_stopped_at is not None:
            required_elapsed_time_after_forced_stop = 1
            current_unix_time = time.time()
            remaining_time = required_elapsed_time_after_forced_stop - \
                (current_unix_time - Global.generation_force_stopped_at)
            if remaining_time > 0:
                time.sleep(remaining_time)
            Global.generation_force_stopped_at = None

        prompter = Prompter(prompt_template)
        prompt = prompter.generate_prompt(list(variables))

        if not prompt:
            return

        generation_config = \
            prepare_generation_config_args(generation_config)
        generation_config = GenerationConfig(
            **generation_config,
        )

        def get_output_for_flagging(output, raw_output, completed=True):
            return json.dumps({
                'model': model_preset.model_name_or_path,
                'adapter_model': model_preset.adapter_model_name_or_path,
                'prompt': prompt,
                'output': output,
                'completed': completed,
                'raw_output': raw_output,
                # 'max_new_tokens': max_new_tokens,
                'generation_config': generation_config.to_dict(),
                'options': {
                    'stop_sequences': stop_sequences,
                },
                'prompt_template': prompt_template,
                'prompt_template_variables': variables,
            })

        def ui_generation_stopping_criteria(input_ids, score, **kwargs):
            if Global.should_stop_generating:
                return True
            return False

        Global.should_stop_generating = False

        model = model_preset.model
        tokenizer = model_preset.tokenizer
        generation_args = {
            'model': model,
            'tokenizer': tokenizer,
            'prompt': prompt,
            'generation_config': generation_config,
            # 'max_new_tokens': max_new_tokens,
            'stopping_criteria': [ui_generation_stopping_criteria],
            'stop_sequences': stop_sequences,
            'skip_special_tokens': model_preset.tokenizer_skip_special_tokens,
            'stream_output': stream_output
        }

        for (
            decoded_output, output, completed
        ) in Global.inference_generate_fn(**generation_args):
            if Global.should_stop_generating:
                return

            response = prompter.get_response(
                output=decoded_output,
                input_variables=list(variables),
            )

            output_tokens_str = comparing_lists(
                [
                    [tokenizer.decode([i]) for i in output],
                    [f"{i}," if c < len(output) - 1 else f"{i}"
                     for c, i in enumerate(output)],
                ],
                labels=['//', ''],
                max_width=52)
            output_tokens_str = output_tokens_str.rstrip()
            output_tokens_str = f"[\n{output_tokens_str}\n]"

            yield (
                gr.Textbox.update(value=response),
                output_tokens_str,
                gr.Textbox.update(
                    value=get_output_for_flagging(
                        decoded_output, output_tokens_str,
                        completed=completed
                    ),
                    visible=True)
            )

            if Global.should_stop_generating:
                # If the user stops the generation, and then clicks the
                # generation button again, they may mysteriously landed
                # here, in the previous, should-be-stopped generation
                # function call, with the new generation function not be
                # called at all. To workaround this, we yield a message
                # and setting lines=1, and if the front-end JS detects
                # that lines has been set to 1 (rows="1" in HTML),
                # it will automatically click the generate button again
                # (gr.Textbox.update() does not support updating
                # elem_classes or elem_id).
                # [WORKAROUND-UI01]
                yield (
                    gr.Textbox.update(
                        value="Please retry", lines=1),
                    None, None)
        return

    except Exception as e:
        raise gr.Error(str(e))


def handle_stop_generate():
    Global.generation_force_stopped_at = time.time()
    Global.should_stop_generating = True
