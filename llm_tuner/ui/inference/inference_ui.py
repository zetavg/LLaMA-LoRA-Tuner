from typing import Any

import gradio as gr
import os
import re
import time
import json
from textwrap import dedent

from transformers import GenerationConfig

from ...config import Config
from ...globals import Global
# from ...models import get_model, get_tokenizer, get_device
from ...lib.csv_logger import CSVLogger
from ...data import (
    get_model_preset_choices,
    # get_available_template_names,
    get_available_lora_model_names,
    get_info_of_available_lora_model,
    get_model_preset_from_choice,
    get_prompt_template_names
)
from ...utils.prompter import Prompter
from ...utils.relative_read_file import relative_read_file

from ..css_styles import register_css_style
from ..components.generation_options import generation_options

from .event_handlers import (
    handle_reload_selections,
    handle_prompt_template_change,
    prepare_generate,
    handle_generate,
    handle_stop_generate,
)

register_css_style('finetune', relative_read_file(__file__, "style.css"))

# device = get_device()

inference_output_lines = 7


class LoggingItem:
    def __init__(self, label):
        self.label = label

    def deserialize(self, value, **kwargs):
        return value


def update_prompt_preview(prompt_template,
                          variable_0, variable_1, variable_2, variable_3,
                          variable_4, variable_5, variable_6, variable_7):
    try:
        variables = [variable_0, variable_1, variable_2, variable_3,
                     variable_4, variable_5, variable_6, variable_7]
        prompter = Prompter(prompt_template)
        prompt = prompter.generate_prompt(variables)
        return prompt
    except Exception as e:
        raise gr.Error(str(e)) from e


def inference_ui():
    flagging_dir = os.path.join(Config.data_dir, "flagging", "inference")
    if not os.path.exists(flagging_dir):
        os.makedirs(flagging_dir)

    flag_callback = CSVLogger()
    flag_components = [
        LoggingItem("Base Model"),
        LoggingItem("Adapter Model"),
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
            config.append(f"Top K: {generation_config.get('top_k')}")
            config.append(f"Top P: {generation_config.get('top_p')}")
        num_beams = generation_config.get('num_beams', 1)
        if num_beams > 1:
            config.append(f"Beams: {generation_config.get('num_beams')}")
        config.append(f"RP: {generation_config.get('repetition_penalty')}")
        return [
            output_for_flagging.get("base_model", ""),
            output_for_flagging.get("adapter_model", ""),
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
        adapter_model = output_for_flagging.get("adapter_model", None)
        if adapter_model == "None":
            adapter_model = None
        if not base_model:
            return "log.csv"
        if not adapter_model:
            return f"log-{base_model}.csv"
        return f"log-{base_model}#{adapter_model}.csv"

    things_that_might_hang = []

    with gr.Blocks() as inference_ui_blocks:
        with gr.Box(elem_classes="form-box disable_while_training"):
            with gr.Row(elem_classes=""):
                model_preset_select = gr.Dropdown(
                    label="Model",
                    elem_id="inference_model_preset_select",
                )
                prompt_template = gr.Dropdown(
                    label="Prompt Template",
                    value="None",
                    elem_id="inference_prompt_template",
                )
            model_prompt_template_message = gr.HTML(
                visible=False,
                elem_classes="mt-m2 ph-2 o-09"
            )
            reload_selections_button = gr.Button(
                "↻", elem_classes="block-reload-btn",
                elem_id="inference_reload_selections_button")
        # with gr.Row(elem_classes="disable_while_training"):
        #     with gr.Column(elem_id="inference_lora_model_group"):
        #         model_prompt_template_message = gr.Markdown(
        #             "", visible=False, elem_id="inference_lora_model_prompt_template_message")
        #         lora_model = gr.Dropdown(
        #             label="LoRA Model",
        #             elem_id="inference_lora_model",
        #             value="None",
        #             allow_custom_value=True,
        #         )
            # prompt_template = gr.Dropdown(
            #     label="Prompt Template",
            #     elem_id="inference_prompt_template",
            # )
            # reload_selections_button = gr.Button(
            #     "↻",
            #     elem_id="inference_reload_selections_button"
            # )
            # reload_selections_button.style(
            #     full_width=False,
            #     size="sm")
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

                    variable_examples = gr.State([])
                    variable_examples_dataset = gr.Dataset(
                        components=[gr.Textbox(visible=False)],
                        samples=[],
                        visible=False,
                        type='index',
                        elem_classes="examples examples-with-width-limit",
                    )
                    variable_examples_select = gr.Dropdown(
                        label="Examples",
                        value='Select an example...',
                        choices=['Select an example...'],
                        visible=False,
                        # type='index',
                        interactive=True,
                        elem_classes="examples",
                    )

                    variable_inputs: Any = [
                        variable_0, variable_1, variable_2, variable_3,
                        variable_4, variable_5, variable_6, variable_7
                    ]

                    def handle_example_choose(i, samples):
                        return_value = []
                        if len(samples) > i:
                            sample = samples[i]
                            return_value = sample
                        while len(return_value) < 8:
                            return_value.append(gr.Textbox.update())
                        return return_value

                    def handle_example_select(s, samples):
                        i = 0
                        m = re.match(f'^[0-9]+', s)
                        if m:
                            i = int(m.group(0)) - 1
                        return handle_example_choose(i, samples)
                    variable_examples_dataset.select(
                        fn=handle_example_choose,
                        inputs=[variable_examples_dataset, variable_examples],
                        outputs=variable_inputs
                    )
                    variable_examples_select.select(
                        fn=handle_example_select,
                        inputs=[variable_examples_select, variable_examples],
                        outputs=variable_inputs
                    ).then(
                        fn=None,
                        inputs=[],
                        outputs=[variable_examples_select],
                        _js=dedent("""
                            function () {
                                setTimeout(function () {
                                    document.querySelector('#inference_variable_0 textarea').focus();
                                }, 200);
                                return ['Select an example...'];
                            }
                            """).strip()
                    )

                    with gr.Accordion("Preview", open=False, elem_id="inference_preview_prompt_container"):
                        preview_prompt = gr.Code(
                            label="Prompt",
                            show_label=False,
                            interactive=False, lines=3,
                            elem_id="inference_preview_prompt")
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
                with gr.Accordion(
                    "Options", open=True,
                    elem_id="inference_options_accordion",
                    elem_classes="gap-0-d2",
                ):
                    go_component = generation_options(
                        elem_id_prefix="inference",
                        elem_classes="bt",
                    )
                    with gr.Row(
                        # elem_id="inference_options_bottom_group"
                        elem_classes="form-btr-0 form-bb-0 form-bl-0 form-br-0"
                    ):
                        stop_sequence = gr.Textbox(
                            label="Stop Sequence",
                            value=Config.default_generation_stop_sequence,
                            # placeholder="Example: 'Human:'",
                            elem_id="inference_stop_sequence",
                        )
                        stream_output = gr.Checkbox(
                            label="Stream Output",
                            elem_id="inference_stream_output",
                            value=True
                        )
                        # show_raw = gr.Checkbox(
                        #     label="Show Raw",
                        #     elem_id="inference_show_raw",
                        #     value=default_show_raw
                        # )

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
                        label="Output",
                        lines=inference_output_lines,
                        interactive=False,
                        elem_id="inference_output")
                    inference_output.style(show_copy_button=True)
                    init_inference_output_btn = gr.Button(
                        visible=False,
                        elem_id="inference_init_inference_output_btn",
                    )
                    init_inference_output_btn.click(
                        fn=lambda x: x,
                        inputs=[inference_output],
                        outputs=[inference_output],
                        # queue=False,
                    )

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
                        "Output Tokens",
                        elem_id="inference_inference_tokens_output_accordion",
                        open=False,
                    ):
                        inference_tokens_output = gr.Code(
                            label="JSON",
                            language="javascript",
                            lines=8,
                            interactive=True,
                            elem_id="inference_tokens_output",
                            elem_classes="cm-max-height-400px")
                        inference_tokens_output_s = gr.Code(
                            visible=False,
                            # label="JSON",
                            # language="javascript",
                            lines=8,
                            interactive=False,
                            elem_id="inference_tokens_output",
                            elem_classes="cm-max-height-400px")
                        inference_tokens_output.change(
                            fn=None,
                            _js="function (v) { return v; }",
                            inputs=[inference_tokens_output_s],
                            outputs=[inference_tokens_output]
                        )

        handle_prompt_template_change_inputs: Any = \
            [
                prompt_template, model_preset_select,
            ]
        handle_prompt_template_change_outputs: Any = \
            [
                model_prompt_template_message,
                variable_examples, variable_examples_dataset, variable_examples_select,
                variable_0, variable_1, variable_2, variable_3,
                variable_4, variable_5, variable_6, variable_7
            ]
        things_that_might_hang.append(
            prompt_template.change(
                fn=handle_prompt_template_change,
                inputs=handle_prompt_template_change_inputs,
                outputs=handle_prompt_template_change_outputs,
                # queue=False,
            )
        )

        def handle_model_preset_select_change(x, y):
            model_preset = get_model_preset_from_choice(x)
            if not model_preset:
                return y

            default_prompt_template = model_preset.default_prompt_template
            if not default_prompt_template or default_prompt_template == 'None':
                return y

            prompt_template_names = get_prompt_template_names()
            if default_prompt_template not in prompt_template_names:
                return y

            return default_prompt_template

        things_that_might_hang.append(
            model_preset_select.change(
                fn=handle_model_preset_select_change,
                # fn=lambda x, y: (
                #     getattr(
                #         get_model_preset_from_choice(x),
                #         'default_prompt_template',
                #         y
                #     )
                # ),
                inputs=[model_preset_select, prompt_template],
                outputs=[prompt_template],
                # queue=False,
            ).then(
                fn=handle_prompt_template_change,
                inputs=handle_prompt_template_change_inputs,
                outputs=handle_prompt_template_change_outputs,
                # queue=False,
            )
        )

        reload_selections_event = reload_selections_button.click(
            handle_reload_selections,
            inputs=[model_preset_select, prompt_template],
            outputs=[model_preset_select, prompt_template],
            # queue=False,
        ).then(
            fn=handle_model_preset_select_change,
            inputs=[model_preset_select, prompt_template],
            outputs=[prompt_template],
            # queue=False,
        ).then(
            fn=handle_prompt_template_change,
            inputs=handle_prompt_template_change_inputs,
            outputs=handle_prompt_template_change_outputs,
            # queue=False,
        )
        things_that_might_hang.append(reload_selections_event)

        # reload_selected_models_btn = gr.Button(
        #     "", elem_id="inference_reload_selected_models_btn")

        # show_raw_change_event = show_raw.change(
        #     fn=lambda show_raw: gr.Accordion.update(visible=show_raw),
        #     inputs=[show_raw],
        #     outputs=[raw_output_group])
        # things_that_might_hang.append(show_raw_change_event)

        # reload_selected_models_btn_event = reload_selected_models_btn.click(
        #     fn=handle_prompt_template_change,
        #     inputs=[prompt_template, lora_model],
        #     outputs=[
        #         model_prompt_template_message,
        #         variable_0, variable_1, variable_2, variable_3, variable_4, variable_5, variable_6, variable_7])
        # things_that_might_hang.append(reload_selected_models_btn_event)

        # lora_model_change_event = lora_model.change(
        #     fn=handle_lora_model_change,
        #     inputs=[lora_model, prompt_template],
        #     outputs=[model_prompt_template_message, prompt_template])
        # things_that_might_hang.append(lora_model_change_event)

        generate_event = generate_btn.click(
            fn=prepare_generate,
            inputs=[model_preset_select],
            outputs=[
                inference_output,
                inference_tokens_output,
                inference_tokens_output_s,
                output_for_flagging
            ],
        ).then(
            fn=handle_generate,
            inputs=[
                model_preset_select,
                prompt_template,
                go_component['generation_config_json'],
                stop_sequence,
                stream_output,
                variable_0, variable_1, variable_2, variable_3,
                variable_4, variable_5, variable_6, variable_7,
            ],
            outputs=[
                inference_output,
                inference_tokens_output,
                inference_tokens_output_s,
                output_for_flagging,
            ],
            api_name="inference"
        )
        stop_btn.click(
            fn=handle_stop_generate,
            inputs=None,
            outputs=None,
            cancels=[generate_event],
            # queue=False,
        )

        update_prompt_preview_event = \
            update_prompt_preview_btn.click(
                fn=update_prompt_preview,
                inputs=[prompt_template,
                        variable_0, variable_1, variable_2, variable_3,
                        variable_4, variable_5, variable_6, variable_7,],
                outputs=preview_prompt,
                postprocess=False,
                # queue=False,
            )
        things_that_might_hang.append(update_prompt_preview_event)

        stop_non_responding_elements_btn = gr.Button(
            "stop non-responding elements",
            elem_classes="foot-stop-non-responding-elements-btn")
        stop_non_responding_elements_btn.click(
            fn=None, inputs=None, outputs=None,
            cancels=things_that_might_hang)

    inference_ui_blocks.load(_js=relative_read_file(__file__, "script.js"))
