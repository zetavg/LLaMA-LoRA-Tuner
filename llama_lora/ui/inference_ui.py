import gradio as gr
import os
import time
import json

from transformers import GenerationConfig

from ..config import Config
from ..globals import Global
from ..models import get_model, get_tokenizer, get_device
from ..lib.csv_logger import CSVLogger
from ..utils.data import (
    get_available_template_names,
    get_available_lora_model_names,
    get_info_of_available_lora_model)
from ..utils.prompter import Prompter

device = get_device()

default_show_raw = True
inference_output_lines = 12


class LoggingItem:
    def __init__(self, label):
        self.label = label

    def deserialize(self, value, **kwargs):
        return value


def prepare_inference(lora_model_name, progress=gr.Progress(track_tqdm=True)):
    base_model_name = Global.base_model_name
    tokenizer_name = Global.tokenizer_name or Global.base_model_name

    try:
        get_tokenizer(tokenizer_name)
        get_model(base_model_name, lora_model_name)
        return ("", "", gr.Textbox.update(visible=False))

    except Exception as e:
        raise gr.Error(e)


def do_inference(
    lora_model_name,
    prompt_template,
    variable_0, variable_1, variable_2, variable_3,
    variable_4, variable_5, variable_6, variable_7,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    repetition_penalty=1.2,
    max_new_tokens=128,
    stream_output=False,
    show_raw=False,
    progress=gr.Progress(track_tqdm=True),
):
    base_model_name = Global.base_model_name

    try:
        if Global.generation_force_stopped_at is not None:
            required_elapsed_time_after_forced_stop = 1
            current_unix_time = time.time()
            remaining_time = required_elapsed_time_after_forced_stop - \
                (current_unix_time - Global.generation_force_stopped_at)
            if remaining_time > 0:
                time.sleep(remaining_time)
            Global.generation_force_stopped_at = None

        variables = [variable_0, variable_1, variable_2, variable_3,
                     variable_4, variable_5, variable_6, variable_7]
        prompter = Prompter(prompt_template)
        prompt = prompter.generate_prompt(variables)

        generation_config = GenerationConfig(
            # to avoid ValueError('`temperature` has to be a strictly positive float, but is 2')
            temperature=float(temperature),
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            num_beams=num_beams,
            # https://github.com/huggingface/transformers/issues/22405#issuecomment-1485527953
            do_sample=temperature > 0,
        )

        def get_output_for_flagging(output, raw_output, completed=True):
            return json.dumps({
                'base_model': base_model_name,
                'adaptor_model': lora_model_name,
                'prompt': prompt,
                'output': output,
                'completed': completed,
                'raw_output': raw_output,
                'max_new_tokens': max_new_tokens,
                'prompt_template': prompt_template,
                'prompt_template_variables': variables,
                'generation_config': generation_config.to_dict(),
            })

        if Config.ui_dev_mode:
            message = f"Hi, I’m currently in UI-development mode and do not have access to resources to process your request. However, this behavior is similar to what will actually happen, so you can try and see how it will work!\n\nBase model: {base_model_name}\nLoRA model: {lora_model_name}\n\nThe following is your prompt:\n\n{prompt}"
            print(message)

            if stream_output:
                def word_generator(sentence):
                    lines = message.split('\n')
                    out = ""
                    for line in lines:
                        words = line.split(' ')
                        for i in range(len(words)):
                            if out:
                                out += ' '
                            out += words[i]
                            yield out
                        out += "\n"
                        yield out

                output = ""
                for partial_sentence in word_generator(message):
                    output = partial_sentence
                    yield (
                        gr.Textbox.update(
                            value=output,
                            lines=inference_output_lines),
                        json.dumps(
                            list(range(len(output.split()))),
                            indent=2),
                        gr.Textbox.update(
                            value=get_output_for_flagging(
                                output, "", completed=False),
                            visible=True)
                    )
                    time.sleep(0.05)

                yield (
                    gr.Textbox.update(
                        value=output,
                        lines=inference_output_lines),
                    json.dumps(
                        list(range(len(output.split()))),
                        indent=2),
                    gr.Textbox.update(
                        value=get_output_for_flagging(
                            output, "", completed=True),
                        visible=True)
                )

                return
            time.sleep(1)
            yield (
                gr.Textbox.update(value=message, lines=inference_output_lines),
                json.dumps(list(range(len(message.split()))), indent=2),
                gr.Textbox.update(
                    value=get_output_for_flagging(message, ""),
                    visible=True)
            )
            return

        tokenizer = get_tokenizer(base_model_name)
        model = get_model(base_model_name, lora_model_name)

        def ui_generation_stopping_criteria(input_ids, score, **kwargs):
            if Global.should_stop_generating:
                return True
            return False

        Global.should_stop_generating = False

        generation_args = {
            'model': model,
            'tokenizer': tokenizer,
            'prompt': prompt,
            'generation_config': generation_config,
            'max_new_tokens': max_new_tokens,
            'stopping_criteria': [ui_generation_stopping_criteria],
            'stream_output': stream_output
        }

        for (decoded_output, output, completed) in Global.inference_generate_fn(**generation_args):
            raw_output_str = str(output)
            response = prompter.get_response(decoded_output)

            if Global.should_stop_generating:
                return

            yield (
                gr.Textbox.update(
                    value=response, lines=inference_output_lines),
                raw_output_str,
                gr.Textbox.update(
                    value=get_output_for_flagging(
                        decoded_output, raw_output_str, completed=completed),
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


def reload_selections(current_lora_model, current_prompt_template):
    available_template_names = get_available_template_names()
    available_template_names_with_none = available_template_names + ["None"]

    if current_prompt_template not in available_template_names_with_none:
        current_prompt_template = None

    current_prompt_template = current_prompt_template or next(
        iter(available_template_names_with_none), None)

    default_lora_models = []
    available_lora_models = default_lora_models + get_available_lora_model_names()
    available_lora_models = available_lora_models + ["None"]

    current_lora_model = current_lora_model or next(
        iter(available_lora_models), None)

    return (gr.Dropdown.update(choices=available_lora_models, value=current_lora_model),
            gr.Dropdown.update(choices=available_template_names_with_none, value=current_prompt_template))


def get_warning_message_for_lora_model_and_prompt_template(lora_model, prompt_template):
    messages = []

    lora_mode_info = get_info_of_available_lora_model(lora_model)

    if lora_mode_info and isinstance(lora_mode_info, dict):
        model_base_model = lora_mode_info.get("base_model")
        if model_base_model and model_base_model != Global.base_model_name:
            messages.append(
                f"⚠️ This model was trained on top of base model `{model_base_model}`, it might not work properly with the selected base model `{Global.base_model_name}`.")

        model_prompt_template = lora_mode_info.get("prompt_template")
        if model_prompt_template and model_prompt_template != prompt_template:
            messages.append(
                f"This model was trained with prompt template `{model_prompt_template}`.")

    return " ".join(messages)


def handle_prompt_template_change(prompt_template, lora_model):
    prompter = Prompter(prompt_template)
    var_names = prompter.get_variable_names()
    human_var_names = [' '.join(word.capitalize()
                                for word in item.split('_')) for item in var_names]
    gr_updates = [gr.Textbox.update(
        label=name, visible=True) for name in human_var_names]
    while len(gr_updates) < 8:
        gr_updates.append(gr.Textbox.update(
            label="Not Used", visible=False))

    model_prompt_template_message_update = gr.Markdown.update(
        "", visible=False)
    warning_message = get_warning_message_for_lora_model_and_prompt_template(
        lora_model, prompt_template)
    if warning_message:
        model_prompt_template_message_update = gr.Markdown.update(
            warning_message, visible=True)

    return [model_prompt_template_message_update] + gr_updates


def handle_lora_model_change(lora_model, prompt_template):
    lora_mode_info = get_info_of_available_lora_model(lora_model)

    if lora_mode_info and isinstance(lora_mode_info, dict):
        model_prompt_template = lora_mode_info.get("prompt_template")
        if model_prompt_template:
            available_template_names = get_available_template_names()
            if model_prompt_template in available_template_names:
                prompt_template = model_prompt_template

    model_prompt_template_message_update = gr.Markdown.update(
        "", visible=False)
    warning_message = get_warning_message_for_lora_model_and_prompt_template(
        lora_model, prompt_template)
    if warning_message:
        model_prompt_template_message_update = gr.Markdown.update(
            warning_message, visible=True)

    return model_prompt_template_message_update, prompt_template


def update_prompt_preview(prompt_template,
                          variable_0, variable_1, variable_2, variable_3,
                          variable_4, variable_5, variable_6, variable_7):
    variables = [variable_0, variable_1, variable_2, variable_3,
                 variable_4, variable_5, variable_6, variable_7]
    prompter = Prompter(prompt_template)
    prompt = prompter.generate_prompt(variables)
    return gr.Textbox.update(value=prompt)


def inference_ui():
    flagging_dir = os.path.join(Config.data_dir, "flagging", "inference")
    if not os.path.exists(flagging_dir):
        os.makedirs(flagging_dir)

    flag_callback = CSVLogger()
    flag_components = [
        LoggingItem("Base Model"),
        LoggingItem("Adaptor Model"),
        LoggingItem("Type"),
        LoggingItem("Prompt"),
        LoggingItem("Output"),
        LoggingItem("Completed"),
        LoggingItem("Config"),
        LoggingItem("Raw Output"),
        LoggingItem("Max New Tokens"),
        LoggingItem("Prompt Template"),
        LoggingItem("Prompt Template Variables"),
        LoggingItem("Generation Config"),
    ]
    flag_callback.setup(flag_components, flagging_dir)

    def get_flag_callback_args(output_for_flagging_str, flag_type):
        output_for_flagging = json.loads(output_for_flagging_str)
        generation_config = output_for_flagging.get("generation_config", {})
        config = []
        if generation_config.get('do_sample', False):
            config.append(
                f"Temperature: {generation_config.get('temperature')}")
            config.append(f"Top P: {generation_config.get('top_p')}")
            config.append(f"Top K: {generation_config.get('top_k')}")
        num_beams = generation_config.get('num_beams', 1)
        if num_beams > 1:
            config.append(f"Beams: {generation_config.get('num_beams')}")
        config.append(f"RP: {generation_config.get('repetition_penalty')}")
        return [
            output_for_flagging.get("base_model", ""),
            output_for_flagging.get("adaptor_model", ""),
            flag_type,
            output_for_flagging.get("prompt", ""),
            output_for_flagging.get("output", ""),
            str(output_for_flagging.get("completed", "")),
            ", ".join(config),
            output_for_flagging.get("raw_output", ""),
            str(output_for_flagging.get("max_new_tokens", "")),
            output_for_flagging.get("prompt_template", ""),
            json.dumps(output_for_flagging.get(
                "prompt_template_variables", "")),
            json.dumps(output_for_flagging.get("generation_config", "")),
        ]

    def get_flag_filename(output_for_flagging_str):
        output_for_flagging = json.loads(output_for_flagging_str)
        base_model = output_for_flagging.get("base_model", None)
        adaptor_model = output_for_flagging.get("adaptor_model", None)
        if adaptor_model == "None":
            adaptor_model = None
        if not base_model:
            return "log.csv"
        if not adaptor_model:
            return f"log-{base_model}.csv"
        return f"log-{base_model}#{adaptor_model}.csv"

    things_that_might_timeout = []

    with gr.Blocks() as inference_ui_blocks:
        with gr.Row(elem_classes="disable_while_training"):
            with gr.Column(elem_id="inference_lora_model_group"):
                model_prompt_template_message = gr.Markdown(
                    "", visible=False, elem_id="inference_lora_model_prompt_template_message")
                lora_model = gr.Dropdown(
                    label="LoRA Model",
                    elem_id="inference_lora_model",
                    value="None",
                    allow_custom_value=True,
                )
            prompt_template = gr.Dropdown(
                label="Prompt Template",
                elem_id="inference_prompt_template",
            )
            reload_selections_button = gr.Button(
                "↻",
                elem_id="inference_reload_selections_button"
            )
            reload_selections_button.style(
                full_width=False,
                size="sm")
        with gr.Row(elem_classes="disable_while_training"):
            with gr.Column():
                with gr.Column(elem_id="inference_prompt_box"):
                    variable_0 = gr.Textbox(
                        lines=2,
                        label="Prompt",
                        placeholder="Tell me about alpecas and llamas.",
                        elem_id="inference_variable_0"
                    )
                    variable_1 = gr.Textbox(
                        lines=2, label="", visible=False, elem_id="inference_variable_1")
                    variable_2 = gr.Textbox(
                        lines=2, label="", visible=False, elem_id="inference_variable_2")
                    variable_3 = gr.Textbox(
                        lines=2, label="", visible=False, elem_id="inference_variable_3")
                    variable_4 = gr.Textbox(
                        lines=2, label="", visible=False, elem_id="inference_variable_4")
                    variable_5 = gr.Textbox(
                        lines=2, label="", visible=False, elem_id="inference_variable_5")
                    variable_6 = gr.Textbox(
                        lines=2, label="", visible=False, elem_id="inference_variable_6")
                    variable_7 = gr.Textbox(
                        lines=2, label="", visible=False, elem_id="inference_variable_7")

                    with gr.Accordion("Preview", open=False, elem_id="inference_preview_prompt_container"):
                        preview_prompt = gr.Textbox(
                            show_label=False, interactive=False, elem_id="inference_preview_prompt")
                        update_prompt_preview_btn = gr.Button(
                            "↻", elem_id="inference_update_prompt_preview_btn")
                        update_prompt_preview_btn.style(size="sm")

                # with gr.Column():
                #     with gr.Row():
                #         generate_btn = gr.Button(
                #             "Generate", variant="primary", label="Generate", elem_id="inference_generate_btn",
                #         )
                #         stop_btn = gr.Button(
                #             "Stop", variant="stop", label="Stop Iterating", elem_id="inference_stop_btn")

                # with gr.Column():
                with gr.Accordion("Options", open=True, elem_id="inference_options_accordion"):
                    temperature = gr.Slider(
                        minimum=0, maximum=2, value=0, step=0.01,
                        label="Temperature",
                        elem_id="inference_temperature"
                    )

                    with gr.Row(elem_classes="inference_options_group"):
                        top_p = gr.Slider(
                            minimum=0, maximum=1, value=0.75, step=0.01,
                            label="Top P",
                            elem_id="inference_top_p"
                        )

                        top_k = gr.Slider(
                            minimum=0, maximum=100, value=40, step=1,
                            label="Top K",
                            elem_id="inference_top_k"
                        )

                    num_beams = gr.Slider(
                        minimum=1, maximum=5, value=2, step=1,
                        label="Beams",
                        elem_id="inference_beams"
                    )

                    repetition_penalty = gr.Slider(
                        minimum=0, maximum=2.5, value=1.2, step=0.01,
                        label="Repetition Penalty",
                        elem_id="inference_repetition_penalty"
                    )

                    max_new_tokens = gr.Slider(
                        minimum=0, maximum=4096, value=128, step=1,
                        label="Max New Tokens",
                        elem_id="inference_max_new_tokens"
                    )

                    with gr.Row(elem_id="inference_options_bottom_group"):
                        stream_output = gr.Checkbox(
                            label="Stream Output",
                            elem_id="inference_stream_output",
                            value=True
                        )
                        show_raw = gr.Checkbox(
                            label="Show Raw",
                            elem_id="inference_show_raw",
                            value=default_show_raw
                        )

                with gr.Column():
                    with gr.Row():
                        generate_btn = gr.Button(
                            "Generate", variant="primary", label="Generate", elem_id="inference_generate_btn",
                        )
                        stop_btn = gr.Button(
                            "Stop", variant="stop", label="Stop Iterating", elem_id="inference_stop_btn")

            with gr.Column(elem_id="inference_output_group_container"):
                with gr.Column(elem_id="inference_output_group"):
                    inference_output = gr.Textbox(
                        lines=inference_output_lines, label="Output", elem_id="inference_output")
                    inference_output.style(show_copy_button=True)

                    with gr.Row(elem_id="inference_flagging_group", variant="panel"):
                        output_for_flagging = gr.Textbox(
                            interactive=False, visible=False,
                            elem_id="inference_output_for_flagging")
                        flag_btn = gr.Button(
                            "Flag", elem_id="inference_flag_btn")
                        flag_up_btn = gr.Button(
                            "👍", elem_id="inference_flag_up_btn")
                        flag_down_btn = gr.Button(
                            "👎", elem_id="inference_flag_down_btn")
                        flag_output = gr.Markdown(
                            "", elem_id="inference_flag_output")
                        flag_btn.click(
                            lambda d: (flag_callback.flag(
                                get_flag_callback_args(d, "Flag"),
                                flag_option="Flag",
                                username=None,
                                filename=get_flag_filename(d)
                            ), "")[1],
                            inputs=[output_for_flagging],
                            outputs=[flag_output],
                            preprocess=False)
                        flag_up_btn.click(
                            lambda d: (flag_callback.flag(
                                get_flag_callback_args(d, "👍"),
                                flag_option="Up Vote",
                                username=None,
                                filename=get_flag_filename(d)
                            ), "")[1],
                            inputs=[output_for_flagging],
                            outputs=[flag_output],
                            preprocess=False)
                        flag_down_btn.click(
                            lambda d: (flag_callback.flag(
                                get_flag_callback_args(d, "👎"),
                                flag_option="Down Vote",
                                username=None,
                                filename=get_flag_filename(d)
                            ), "")[1],
                            inputs=[output_for_flagging],
                            outputs=[flag_output],
                            preprocess=False)

                    with gr.Accordion(
                            "Raw Output",
                            open=not default_show_raw,
                            visible=default_show_raw,
                            elem_id="inference_inference_raw_output_accordion"
                    ) as raw_output_group:
                        inference_raw_output = gr.Code(
                            # label="Raw Output",
                            label="Tensor",
                            language="json",
                            lines=8,
                            interactive=False,
                            elem_id="inference_raw_output")

        reload_selected_models_btn = gr.Button(
            "", elem_id="inference_reload_selected_models_btn")

        show_raw_change_event = show_raw.change(
            fn=lambda show_raw: gr.Accordion.update(visible=show_raw),
            inputs=[show_raw],
            outputs=[raw_output_group])
        things_that_might_timeout.append(show_raw_change_event)

        reload_selections_event = reload_selections_button.click(
            reload_selections,
            inputs=[lora_model, prompt_template],
            outputs=[lora_model, prompt_template],
        )
        things_that_might_timeout.append(reload_selections_event)

        prompt_template_change_event = prompt_template.change(
            fn=handle_prompt_template_change,
            inputs=[prompt_template, lora_model],
            outputs=[
                model_prompt_template_message,
                variable_0, variable_1, variable_2, variable_3, variable_4, variable_5, variable_6, variable_7])
        things_that_might_timeout.append(prompt_template_change_event)

        reload_selected_models_btn_event = reload_selected_models_btn.click(
            fn=handle_prompt_template_change,
            inputs=[prompt_template, lora_model],
            outputs=[
                model_prompt_template_message,
                variable_0, variable_1, variable_2, variable_3, variable_4, variable_5, variable_6, variable_7])
        things_that_might_timeout.append(reload_selected_models_btn_event)

        lora_model_change_event = lora_model.change(
            fn=handle_lora_model_change,
            inputs=[lora_model, prompt_template],
            outputs=[model_prompt_template_message, prompt_template])
        things_that_might_timeout.append(lora_model_change_event)

        generate_event = generate_btn.click(
            fn=prepare_inference,
            inputs=[lora_model],
            outputs=[inference_output,
                     inference_raw_output, output_for_flagging],
        ).then(
            fn=do_inference,
            inputs=[
                lora_model,
                prompt_template,
                variable_0, variable_1, variable_2, variable_3,
                variable_4, variable_5, variable_6, variable_7,
                temperature,
                top_p,
                top_k,
                num_beams,
                repetition_penalty,
                max_new_tokens,
                stream_output,
                show_raw,
            ],
            outputs=[inference_output,
                     inference_raw_output, output_for_flagging],
            api_name="inference"
        )
        stop_btn.click(
            fn=handle_stop_generate,
            inputs=None,
            outputs=None,
            cancels=[generate_event]
        )

        update_prompt_preview_event = update_prompt_preview_btn.click(fn=update_prompt_preview, inputs=[prompt_template,
                                                                                                        variable_0, variable_1, variable_2, variable_3,
                                                                                                        variable_4, variable_5, variable_6, variable_7,], outputs=preview_prompt)
        things_that_might_timeout.append(update_prompt_preview_event)

        stop_timeoutable_btn = gr.Button(
            "stop not-responding elements",
            elem_id="inference_stop_timeoutable_btn",
            elem_classes="foot_stop_timeoutable_btn")
        stop_timeoutable_btn.click(
            fn=None, inputs=None, outputs=None, cancels=things_that_might_timeout)

    inference_ui_blocks.load(_js="""
    function inference_ui_blocks_js() {
      // Auto load options
      setTimeout(function () {
        document.getElementById('inference_reload_selections_button').click();

        // Workaround default value not shown.
        document.querySelector('#inference_lora_model input').value =
          'None';
      }, 100);

      // Add tooltips
      setTimeout(function () {
        tippy('#inference_lora_model', {
          placement: 'top-start',
          delay: [500, 0],
          animation: 'scale-subtle',
          content:
            'Select a LoRA model form your data directory, or type in a model name on HF (e.g.: <code>tloen/alpaca-lora-7b</code>).',
          allowHTML: true,
        });

        tippy('#inference_prompt_template', {
          placement: 'top-start',
          delay: [500, 0],
          animation: 'scale-subtle',
          content:
            'Templates are loaded from the "templates" folder of your data directory. Be sure to select the template that matches your selected LoRA model to get the best results.',
        });

        tippy('#inference_reload_selections_button', {
          placement: 'bottom-end',
          delay: [500, 0],
          animation: 'scale-subtle',
          content: 'Press to reload LoRA Model and Prompt Template selections.',
        });

        document
          .querySelector('#inference_preview_prompt_container .label-wrap')
          .addEventListener('click', function () {
            tippy('#inference_preview_prompt', {
              placement: 'right',
              delay: [500, 0],
              animation: 'scale-subtle',
              content: 'This is the prompt that will be sent to the language model.',
            });

            const update_btn = document.getElementById(
              'inference_update_prompt_preview_btn'
            );
            if (update_btn) update_btn.click();
          });

        function setTooltipForOptions() {
          tippy('#inference_temperature', {
            placement: 'right',
            delay: [500, 0],
            animation: 'scale-subtle',
            content:
              '<strong>Controls randomness</strong>: Higher values (e.g., <code>1.0</code>) make the model generate more diverse and random outputs. As the temperature approaches zero, the model will become deterministic and repetitive.<br /><i>Setting a value larger then <code>0</code> will enable sampling.</i>',
            allowHTML: true,
          });

          tippy('#inference_top_p', {
            placement: 'right',
            delay: [500, 0],
            animation: 'scale-subtle',
            content:
              'Controls diversity via nucleus sampling: only the tokens whose cumulative probability exceeds <code>top_p</code> are considered. <code>0.5</code> means half of all likelihood-weighted options are considered.<br />Will only take effect if Temperature is set to > 0.',
            allowHTML: true,
          });

          tippy('#inference_top_k', {
            placement: 'right',
            delay: [500, 0],
            animation: 'scale-subtle',
            content:
              'Controls diversity of the generated text by only considering the <code>top_k</code> tokens with the highest probabilities. This method can lead to more focused and coherent outputs by reducing the impact of low probability tokens.<br />Will only take effect if Temperature is set to > 0.',
            allowHTML: true,
          });

          tippy('#inference_beams', {
            placement: 'right',
            delay: [500, 0],
            animation: 'scale-subtle',
            content:
              'Number of candidate sequences explored in parallel during text generation using beam search. A higher value increases the chances of finding high-quality, coherent output, but may slow down the generation process.',
          });

          tippy('#inference_repetition_penalty', {
            placement: 'right',
            delay: [500, 0],
            animation: 'scale-subtle',
            content:
              'Applies a penalty to the probability of tokens that have already been generated, discouraging the model from repeating the same words or phrases. The penalty is applied by dividing the token probability by a factor based on the number of times the token has appeared in the generated text.',
          });

          tippy('#inference_max_new_tokens', {
            placement: 'right',
            delay: [500, 0],
            animation: 'scale-subtle',
            content:
              'Limits the maximum number of tokens generated in a single iteration.',
          });

          tippy('#inference_stream_output', {
            placement: 'right',
            delay: [500, 0],
            animation: 'scale-subtle',
            content:
              'When enabled, generated text will be displayed in real-time as it is being produced by the model, allowing you to observe the text generation process as it unfolds.',
          });
        }
        setTooltipForOptions();

        const inference_options_accordion_toggle = document.querySelector(
          '#inference_options_accordion .label-wrap'
        );
        if (inference_options_accordion_toggle) {
          inference_options_accordion_toggle.addEventListener('click', function () {
            setTooltipForOptions();
          });
        }
      }, 100);

      // Show/hide generate and stop button base on the state.
      setTimeout(function () {
        // Make the '#inference_output > .wrap' element appear
        document.getElementById('inference_stop_btn').click();

        setTimeout(function () {
          const output_wrap_element = document.querySelector(
            '#inference_output > .wrap'
          );
          function handle_output_wrap_element_class_change() {
            if (Array.from(output_wrap_element.classList).includes('hide')) {
              document.getElementById('inference_generate_btn').style.display =
                'block';
              document.getElementById('inference_stop_btn').style.display = 'none';
            } else {
              document.getElementById('inference_generate_btn').style.display =
                'none';
              document.getElementById('inference_stop_btn').style.display = 'block';
            }
          }
          new MutationObserver(function (mutationsList, observer) {
            handle_output_wrap_element_class_change();
          }).observe(output_wrap_element, {
            attributes: true,
            attributeFilter: ['class'],
          });
          handle_output_wrap_element_class_change();
        }, 500);
      }, 0);

      // Reload model selection on possible base model change.
      setTimeout(function () {
        const elem = document.getElementById('main_page_tabs_container');
        if (!elem) return;

        let prevClassList = [];

        new MutationObserver(function (mutationsList, observer) {
          const currentPrevClassList = prevClassList;
          const currentClassList = Array.from(elem.classList);
          prevClassList = Array.from(elem.classList);

          if (!currentPrevClassList.includes('hide')) return;
          if (currentClassList.includes('hide')) return;

          const inference_reload_selected_models_btn_elem = document.getElementById('inference_reload_selected_models_btn');

          if (inference_reload_selected_models_btn_elem) inference_reload_selected_models_btn_elem.click();
        }).observe(elem, {
          attributes: true,
          attributeFilter: ['class'],
        });
      }, 0);

      // Debounced updating the prompt preview.
      setTimeout(function () {
        function debounce(func, wait) {
          let timeout;
          return function (...args) {
            const context = this;
            clearTimeout(timeout);
            const fn = () => {
              if (document.querySelector('#inference_preview_prompt > .wrap:not(.hide)')) {
                // Preview request is still loading, wait for 10ms and try again.
                timeout = setTimeout(fn, 10);
                return;
              }
              func.apply(context, args);
            };
            timeout = setTimeout(fn, wait);
          };
        }

        function update_preview() {
          const update_btn = document.getElementById(
            'inference_update_prompt_preview_btn'
          );
          if (!update_btn) return;

          update_btn.click();
        }

        for (let i = 0; i < 8; i++) {
          const e = document.querySelector(`#inference_variable_${i} textarea`);
          if (!e) return;
          e.addEventListener('input', debounce(update_preview, 500));
        }

        const prompt_template_selector = document.querySelector(
          '#inference_prompt_template .wrap-inner'
        );

        if (prompt_template_selector) {
          new MutationObserver(
            debounce(function () {
              if (prompt_template_selector.classList.contains('showOptions')) return;
              update_preview();
            }, 500)
          ).observe(prompt_template_selector, {
            attributes: true,
            attributeFilter: ['class'],
          });
        }
      }, 100);

      // [WORKAROUND-UI01]
      setTimeout(function () {
        const inference_output_textarea = document.querySelector(
          '#inference_output textarea'
        );
        if (!inference_output_textarea) return;
        const observer = new MutationObserver(function () {
          if (inference_output_textarea.getAttribute('rows') === '1') {
            setTimeout(function () {
              const inference_generate_btn = document.getElementById(
                'inference_generate_btn'
              );
              if (inference_generate_btn) inference_generate_btn.click();
            }, 10);
          }
        });
        observer.observe(inference_output_textarea, {
          attributes: true,
          attributeFilter: ['rows'],
        });
      }, 100);

      return [];
    }
    """)
