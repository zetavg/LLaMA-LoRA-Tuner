from typing import Any

import json
from textwrap import dedent
import gradio as gr
from transformers import TRANSFORMERS_CACHE

from ....config import Config
from ....data import get_available_template_names
from ...ui_utils import mk_fake_btn, tie_controls_with_json_editor

from .event_handlers import (
    handle_load_model_presets,
    handle_show_model_preset,
    handle_new_model_preset,
    handle_edit_model_preset,
    handle_duplicate_model_preset,
    handle_delete_model_preset,
    handle_set_model_preset_as_default,
    handle_toggle_model_preset_star,
    handle_save_edit,
    handle_discard_edit,
)


def model_presets_ui():
    things_that_might_hang = []
    with gr.Blocks():
        with gr.Column() as main_view:
            gr.Markdown("""
                Define model presets to switch between different models.
            """, elem_classes="info-text-color")
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Column(elem_classes="model-preset-list"):
                        with gr.Row(elem_classes="gap-half"):
                            new_button = gr.Button(
                                "+ New",
                                elem_classes="mw-fc flex-1",
                            )
                            reload_button = gr.Button(
                                "↻ Reload",
                                elem_id="models_preset_list_reload_button",
                                elem_classes="mw-fc flex-2",
                            )
                        model_presets_list = gr.HTML()
                        model_presets_list_status = gr.HTML(visible=False)
                        load_model_presets_outputs: Any = [
                            model_presets_list,
                            model_presets_list_status,
                        ]
                        things_that_might_hang.append(
                            reload_button.click(
                                fn=handle_load_model_presets,
                                inputs=[],
                                outputs=load_model_presets_outputs
                            )
                        )

                    with gr.Column(visible=False):
                        with gr.Row(
                                variant="panel",
                                elem_classes="panel-with-textbox-and-btn"):
                            model_preset_uid_to_show = gr.Textbox(
                                label="Preset UID to Show", lines=1, max_lines=1,
                                elem_id="models_model_preset_uid_to_show"
                            )
                            show_model_preset_btn = gr.Button(
                                "Show",
                                elem_id="models_show_model_preset_btn"
                            )
                        with gr.Row(
                                variant="panel",
                                elem_classes="panel-with-textbox-and-btn"):
                            model_preset_uid_to_edit = gr.Textbox(
                                label="Preset UID to Edit", lines=1, max_lines=1,
                                elem_id="models_model_preset_uid_to_edit"
                            )
                            edit_model_preset_btn = gr.Button(
                                "Edit",
                                elem_id="models_edit_model_preset_btn"
                            )
                        with gr.Row(
                                variant="panel",
                                elem_classes="panel-with-textbox-and-btn"):
                            model_preset_uid_to_duplicate = gr.Textbox(
                                label="Preset UID to Duplicate", lines=1, max_lines=1,
                                elem_id="models_model_preset_uid_to_duplicate"
                            )
                            duplicate_model_preset_btn = gr.Button(
                                "Duplicate",
                                elem_id="models_duplicate_model_preset_btn"
                            )
                        with gr.Row(
                                variant="panel",
                                elem_classes="panel-with-textbox-and-btn"):
                            model_preset_uid_to_delete = gr.Textbox(
                                label="Preset UID to Delete", lines=1, max_lines=1,
                                elem_id="models_model_preset_uid_to_delete"
                            )
                            delete_model_preset_btn = gr.Button(
                                "Delete",
                                elem_id="models_delete_model_preset_btn"
                            )
                        with gr.Row(
                                variant="panel",
                                elem_classes="panel-with-textbox-and-btn"):
                            model_preset_uid_to_set_as_default = gr.Textbox(
                                label="Preset UID to Set as Default", lines=1, max_lines=1,
                                elem_id="models_model_preset_uid_to_set_as_default"
                            )
                            set_model_preset_as_default_btn = gr.Button(
                                "Set as Default",
                                elem_id="models_set_as_default_model_preset_btn"
                            )
                        with gr.Row(
                                variant="panel",
                                elem_classes="panel-with-textbox-and-btn"):
                            model_preset_uid_to_toggle_star = gr.Textbox(
                                label="Preset UID to Toggle Star", lines=1, max_lines=1,
                                elem_id="models_model_preset_uid_to_toggle_star"
                            )
                            toggle_model_preset_star_btn = gr.Button(
                                "Toggle Star",
                                elem_id="models_toggle_model_preset_star_btn"
                            )

                with gr.Column(scale=2,
                               elem_id="models_right_column_container",
                               elem_classes="models_right_column_container"):
                    with gr.Column(
                        visible=True, variant='panel',
                        elem_classes='models-ui-empty',
                    ) as right_column_empty:
                        gr.Markdown(
                            "Select a preset to view details", visible=True)
                    with gr.Column(
                            visible=False, variant='panel',
                            elem_classes="gap-0"
                    ) as right_column_show:
                        rcs_title = gr.HTML('', elem_classes="mb-md")
                        rcs_json = gr.Code(label="JSON", language='json')

            show_model_preset_outputs = [
                right_column_empty,  # type: ignore
                right_column_show,  # type: ignore
                rcs_title,
                rcs_json,
            ]
            things_that_might_hang.append(
                show_model_preset_btn.click(
                    fn=handle_show_model_preset,
                    inputs=[model_preset_uid_to_show],
                    outputs=show_model_preset_outputs
                )
            )

        with gr.Box(visible=False, elem_classes="t-bg max-width-800") as edit_view:
            with gr.Column(elem_classes="mb-xxl gap-0"):
                with gr.Row(elem_classes="mb-md"):
                    ev_title = gr.HTML('', elem_classes="display-flex ai-c")
                    with gr.Row(elem_classes="models-ui-box-actions"):
                        discard_edit_btn = gr.Button(
                            "Confirm Discard", variant='stop',
                            visible=False,
                            elem_id="models_discard_edit_btn"
                        )
                        fake_discard_edit_btn = gr.Button(
                            "Discard",
                            elem_id="models_fake_discard_edit_btn")
                        mk_fake_btn(
                            fake_discard_edit_btn,
                            elem_id="models_fake_discard_edit_btn",
                            real_elem_id="models_discard_edit_btn"
                        )
                        save_edit_btn = gr.Button("Save", variant='primary')
                save_edit_message = gr.HTML(
                    visible=False, elem_classes="models-ui-save-message")
            with gr.Column(visible=False):
                ev_uid = gr.Textbox()
                ev_original_file_name = gr.Textbox()
            gr.Markdown("### Basic Info", elem_classes="mb-md")
            with gr.Box(elem_classes="form-box"):
                ev_preset_name = gr.Textbox(
                    label="Preset Name",
                    lines=1, max_lines=1)
                gr.Markdown(
                    elem_classes='info-text mt-m-md ph-md',
                    value=dedent(f"""
                       Name the preset for easy recolonization.
                    """).strip()
                )
            gr.Markdown("### Model", elem_classes="mt-xxl mb-md")
            with gr.Box(elem_classes="form-box"):
                with gr.Row(elem_classes="deep-gap-block-padding"):
                    ev_model_from_default_value = 'HF Hub'
                    ev_model_from = gr.Dropdown(
                        label="From",
                        choices=[
                            'Data Dir',
                            'HF Hub',
                        ],
                        value=ev_model_from_default_value,
                        elem_classes="flex-grow-0"
                    )
                    ev_model_name = gr.Textbox(
                        label="Model Name",
                        placeholder="Example: huggyllama/llama-7b",
                        lines=1, max_lines=1,
                    )
                    ev_model_name_select = gr.Dropdown(
                        visible=False,
                        label="Model",
                        choices=[
                            'TODO',
                            'WIP',
                        ]
                    )
                    ev_model_from.change(
                        fn=lambda v: (
                            gr.Textbox.update(visible=v == 'HF Hub'),
                            gr.Dropdown.update(visible=v == 'Data Dir')
                        ),
                        inputs=[ev_model_from],
                        outputs=[ev_model_name, ev_model_name_select]
                    )
                model_from_info_text_mapping = {
                    'HF Hub': f'The model will be automatically downloaded and cached by the HF Transformers library (at <code>{TRANSFORMERS_CACHE}</code>).',
                    'Data Dir': f'Select a model from the local data dir (<code>{Config.models_path}</code>).',
                }
                model_from_info_text = gr.Markdown(
                    elem_classes='info-text mt-m-md ph-md',
                    value=model_from_info_text_mapping[
                        ev_model_from_default_value
                    ]
                )
                ev_model_from.change(
                    fn=None,
                    inputs=[ev_model_from],
                    outputs=[model_from_info_text],
                    _js=f"function (k) {{ return {json.dumps(model_from_info_text_mapping)}[k] }}",
                )

                with gr.Row(elem_classes="deep-gap-block-padding"):
                    with gr.Column():
                        ev_model_torch_dtype = gr.Dropdown(
                            label="Torch dtype",
                            choices=[
                                'Default',
                                'auto',
                                'float16',
                                'bfloat16',
                                'float32',
                                'float64',
                                'complex64',
                                'complex128',
                            ],
                            value="Default"
                        )
                        ev_model_torch_dtype_value_mapping = {
                            'Default': 'undefined',
                            'auto': '"auto"',
                            'float16': '"float16"',
                            'bfloat16': '"bfloat16"',
                            'float32': '"float32"',
                            'float64': '"float64"',
                            'complex64': '"complex64"',
                            'complex128': '"complex128"',
                        }
                        gr.Markdown(
                            elem_classes='info-text mt-m-md ph-md',
                            value=dedent(f"""
                                Choice "Default" to use the global config (`{Config.torch_dtype}`).
                                For more information, check the docs [here](https://huggingface.co/docs/transformers/v4.28.1/en/main_classes/model#model-instantiation-dtype).
                            """).strip()
                        )
                    with gr.Column():
                        ev_model_load_in_8bit = gr.Dropdown(
                            label="Load in 8bit",
                            choices=['Yes', 'No', 'Default'],
                            allow_custom_value=False,
                        )
                        ev_model_load_in_8bit_value_mapping = {
                            'Yes': 'true',
                            'No': 'false',
                            'Default': 'undefined',
                        }
                        gr.Markdown(
                            elem_classes='info-text mt-m-md ph-md',
                            value=dedent(f"""
                                Whether to load the model in 8-bit quantization. Choice "Default" to use the global config (`{'Yes' if Config.load_in_8bit else 'No'}`).
                                For more information, check the docs [here](https://huggingface.co/docs/transformers/v4.28.1/main_classes/quantization#load-a-large-model-in-8bit).
                            """).strip()
                        )
                ev_model_trust_remote_code = gr.Checkbox(
                    label="Trust Remote Code"
                )
            with gr.Box(elem_classes="form-box deep-gap-block-padding mt-gap"):
                ev_use_adapter_model = gr.Checkbox(
                    label="Use Adapter Model (LoRA)",
                    value=False,
                )
                gr.Markdown(
                    elem_classes='info-text mt-m-md ph-md',
                    value=dedent(f"""
                       Apply an adapter (LoRA) model.
                    """).strip()
                )
                with gr.Column(visible=False) as ev_adapter_model_settings:
                    with gr.Row():
                        ev_adapter_model_from_default_value = "HF Hub"
                        ev_adapter_model_from = gr.Dropdown(
                            label="From",
                            choices=[
                                'Data Dir',
                                'HF Hub',
                            ],
                            value=ev_adapter_model_from_default_value,
                            elem_classes="flex-grow-0"
                        )
                        ev_adapter_model_name = gr.Textbox(
                            label="Adapter (LoRA) Model Name",
                            placeholder="Example: tloen/alpaca-lora-7b",
                            lines=1, max_lines=1)
                        ev_adapter_model_name_select = gr.Dropdown(
                            visible=False,
                            label="Adapter (LoRA) Model",
                            choices=[
                                'TODO',
                                'WIP',
                            ]
                        )
                        ev_adapter_model_from.change(
                            fn=lambda v: (
                                gr.Textbox.update(visible=v == 'HF Hub'),
                                gr.Dropdown.update(visible=v == 'Data Dir')
                            ),
                            inputs=[ev_adapter_model_from],
                            outputs=[ev_adapter_model_name,
                                     ev_adapter_model_name_select]
                        )
                    adapter_model_from_info_text_mapping = {
                        'HF Hub': f'The model will be automatically downloaded and cached by the HF Transformers library (at <code>{TRANSFORMERS_CACHE}</code>).',
                        'Data Dir': f'Select a model from the local data dir (<code>{Config.adapter_models_path}</code>).',
                    }
                    adapter_model_from_info_text = gr.Markdown(
                        elem_classes='info-text mt-m-md ph-md',
                        value=adapter_model_from_info_text_mapping[
                            ev_adapter_model_from_default_value
                        ]
                    )
                    ev_adapter_model_from.change(
                        fn=None,
                        inputs=[ev_adapter_model_from],
                        outputs=[adapter_model_from_info_text],
                        _js=f"function (k) {{ return {json.dumps(adapter_model_from_info_text_mapping)}[k] }}",
                    )
                ev_use_adapter_model.change(
                    fn=lambda enabled: gr.Column.update(visible=enabled),
                    inputs=[ev_use_adapter_model],
                    outputs=[ev_adapter_model_settings],  # type: ignore
                )

            gr.Markdown("### Defaults", elem_classes="mt-xxl mb-m8")
            with gr.Box(elem_classes="form-box mt-gap"):
                with gr.Row(elem_classes="deep-gap-block-padding"):
                    ev_default_prompt_template_select = gr.Dropdown(
                        label="Default Prompt Template",
                        choices=['None'],
                        value='None'
                    )
                    reload_model_defaults_selections_button = gr.Button(
                        "↻", elem_classes="block-reload-btn",
                        elem_id="model_presets_reload_model_defaults_selections_button"
                    )
                    reload_model_defaults_selections_button.click(
                        fn=lambda: (
                            gr.Dropdown.update(
                                choices=get_available_template_names() +
                                ['None'],
                            )
                        ),
                        inputs=[],
                        outputs=[ev_default_prompt_template_select]
                    )
                gr.Markdown(
                    elem_classes='info-text mt-m-md ph-md',
                    value=dedent(f"""
                        Select a default prompt template while using this preset.
                    """).strip()
                )
            gr.Markdown("### Advanced", elem_classes="mt-xxl mb-m8")
            with gr.Box(elem_classes="form-box mt-gap"):
                ev_use_custom_tokenizer = gr.Checkbox(
                    label="Use Custom Tokenizer",
                    value=False,
                )
                gr.Markdown(
                    elem_classes='info-text mt-m-md ph-md',
                    value=dedent(f"""
                       Use a tokenizer that is not shipped with the model.
                    """).strip()
                )
                with gr.Column(visible=False) as ev_custom_tokenizer_settings:
                    with gr.Row(elem_classes="deep-gap-block-padding"):
                        ev_custom_tokenizer_from = gr.Dropdown(
                            label="From",
                            choices=[
                                'Data Dir',
                                'HF Hub',
                            ],
                            value="HF Hub",
                            elem_classes="flex-grow-0"
                        )
                        ev_custom_tokenizer_from_value_mapping = {
                            'Data Dir': '"data_dir"',
                            'HF Hub': '"default"',
                        }
                        ev_custom_tokenizer_name = gr.Textbox(
                            label="Tokenizer Name",
                            lines=1, max_lines=1)
                        ev_custom_tokenizer_name_select = gr.Dropdown(
                            visible=False,
                            label="Tokenizer",
                            choices=[
                                'TODO',
                                'WIP',
                            ]
                        )
                        ev_custom_tokenizer_from.change(
                            fn=lambda v: (
                                gr.Textbox.update(visible=v == 'HF Hub'),
                                gr.Dropdown.update(visible=v == 'Data Dir')
                            ),
                            inputs=[ev_custom_tokenizer_from],
                            outputs=[ev_custom_tokenizer_name,
                                     ev_custom_tokenizer_name_select]
                        )
                ev_use_custom_tokenizer.change(
                    fn=lambda enabled: gr.Column.update(visible=enabled),
                    inputs=[ev_use_custom_tokenizer],
                    outputs=[ev_custom_tokenizer_settings],  # type: ignore
                )
            with gr.Accordion(
                label="Model Preset JSON Editor",
                open=False,
                elem_id="models_edit_advanced_accordion",
                elem_classes="accordion deep-gap-0 mt-gap",
            ):
                ev_json = gr.Code(label="JSON", language='json')
                ev_json_message = gr.HTML('')
                gr.Markdown(
                    elem_classes='info-text mt-sm ph-md',
                    value=dedent(f"""
                        Control everything that is unadjustable by the UI.

                        &nbsp;&nbsp;•&nbsp; `.model.args` is the arguments that will be passed into the `.from_pretrained()` function while loading the model ([docs](https://huggingface.co/docs/transformers/v4.28.1/en/model_doc/auto#transformers.AutoModelForCausalLM.from_pretrained)).<br/>
                        &nbsp;&nbsp;•&nbsp; `.adapter_model` will not be used if `.use_adapter_model` is set to `false`.
                    """).strip()
                )
                # with gr.Column(variant="panel"):
                #     gr.Markdown(dedent("""
                #         **💡 Tips**

                #         * The model will be loaded as
                #             ```python
                #             AutoModelForCausalLM.from_pretrained(
                #                 json['model']['name_or_path'],
                #                 **json['model']['args']
                #             )
                #             ```
                #     """).strip())

            tie_controls_with_json_editor(
                ev_json,
                [
                    (ev_preset_name, 'name', 'string'),
                    (ev_model_name, ['model', 'name_or_path'], 'string'),
                    (ev_model_name_select,
                        ['model', 'name_or_path'], 'string'),
                    (ev_model_from, ['model', 'load_from'],
                        {'Data Dir': '"data_dir"', 'HF Hub': '"default"'}),
                    (ev_model_torch_dtype,
                        ['model', 'args', 'torch_dtype'],
                        ev_model_torch_dtype_value_mapping),
                    (ev_model_load_in_8bit,
                        ['model', 'args', 'load_in_8bit'],
                        ev_model_load_in_8bit_value_mapping),
                    (ev_model_trust_remote_code,
                        ['model', 'args', 'trust_remote_code'], 'boolean'),
                    (ev_use_custom_tokenizer,
                        ['use_custom_tokenizer'], 'boolean'),
                    (ev_custom_tokenizer_name,
                        ['custom_tokenizer', 'name_or_path'], 'string'),
                    (ev_custom_tokenizer_name_select, [
                     'custom_tokenizer', 'name_or_path'], 'string'),
                    (ev_custom_tokenizer_from,
                        ['custom_tokenizer', 'load_from'],
                        ev_custom_tokenizer_from_value_mapping),
                    (ev_use_adapter_model,
                        ['use_adapter_model'], 'boolean'),
                    (ev_adapter_model_name,
                        ['adapter_model', 'name_or_path'], 'string'),
                    (ev_adapter_model_name_select, [
                     'adapter_model', 'name_or_path'], 'string'),
                    (ev_adapter_model_from, ['adapter_model', 'load_from'],
                        {'Data Dir': '"data_dir"', 'HF Hub': '"default"'}),
                    (ev_default_prompt_template_select,
                        ['defaults', 'prompt_template'], 'string'),
                ],
                ev_json_message,
                status_indicator_elem_id="models_edit_advanced_accordion",
            )

        stop_timeoutable_btn = gr.Button(
            "stop not-responding elements",
            elem_id="inference_stop_timeoutable_btn",
            elem_classes="foot_stop_timeoutable_btn")
        stop_timeoutable_btn.click(
            fn=None, inputs=None, outputs=None, cancels=things_that_might_hang)

        enter_editing_outputs: Any = [
            main_view,
            edit_view,
            ev_title,
            save_edit_message,
            ev_uid,
            ev_original_file_name,
            ev_json,
        ]
        things_that_might_hang.append(
            new_button.click(
                fn=handle_new_model_preset,
                inputs=[],
                outputs=enter_editing_outputs)
        )
        things_that_might_hang.append(
            edit_model_preset_btn.click(
                fn=handle_edit_model_preset,
                inputs=[model_preset_uid_to_edit],
                outputs=enter_editing_outputs
            )
        )
        things_that_might_hang.append(
            duplicate_model_preset_btn.click(
                fn=handle_duplicate_model_preset,
                inputs=[model_preset_uid_to_duplicate],
                outputs=enter_editing_outputs
            )
        )
        things_that_might_hang.append(
            delete_model_preset_btn.click(
                fn=handle_delete_model_preset,
                inputs=[model_preset_uid_to_delete],
                outputs=load_model_presets_outputs + [
                    main_view,  # type: ignore
                    right_column_empty,   # type: ignore
                    edit_view,  # type: ignore
                    right_column_show,   # type: ignore
                ]
            )
        )
        things_that_might_hang.append(
            set_model_preset_as_default_btn.click(
                fn=handle_set_model_preset_as_default,
                inputs=[model_preset_uid_to_set_as_default],
                outputs=load_model_presets_outputs + show_model_preset_outputs
            )
        )
        things_that_might_hang.append(
            toggle_model_preset_star_btn.click(
                fn=handle_toggle_model_preset_star,
                inputs=[model_preset_uid_to_toggle_star],
                outputs=load_model_presets_outputs + show_model_preset_outputs
            )
        )
        things_that_might_hang.append(
            save_edit_btn.click(
                fn=handle_save_edit,
                inputs=[ev_uid, ev_original_file_name, ev_json],
                outputs=[
                    save_edit_message,
                    model_preset_uid_to_show,
                    main_view,  # type: ignore
                    edit_view,  # type: ignore
                ]
            ).then(
                fn=handle_show_model_preset,
                inputs=[model_preset_uid_to_show],
                outputs=[
                    right_column_empty,  # type: ignore
                    right_column_show,  # type: ignore
                    rcs_title,
                    rcs_json,
                ]
            ).then(
                fn=handle_load_model_presets,
                inputs=[],
                outputs=load_model_presets_outputs,
            )
        )
        things_that_might_hang.append(
            discard_edit_btn.click(
                fn=handle_discard_edit,
                inputs=[],
                outputs=[
                    main_view,  # type: ignore
                    edit_view,  # type: ignore
                ]
            )
        )
