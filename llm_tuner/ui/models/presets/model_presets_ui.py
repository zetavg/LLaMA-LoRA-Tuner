from typing import Any

import gradio as gr

from ...ui_utils import mk_fake_btn, tie_controls_with_json_editor

from .event_handlers import (
    handle_load_model_presets,
    handle_show_model_preset,
    handle_new_model_preset,
    handle_edit_model_preset,
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
                Define model presets to switch between different models and configurations.
            """)
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

        with gr.Box(visible=False, elem_classes="t-bg") as edit_view:
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
            ev_preset_name = gr.Textbox(
                label="Preset Name",
                lines=1, max_lines=1)
            gr.Markdown("### Model", elem_classes="mt-xxl mb-md")
            with gr.Box(elem_classes="form-box"):
                with gr.Row(elem_classes="gap-block-padding"):
                    ev_model_from = gr.Dropdown(
                        label="From",
                        choices=[
                            'Data Dir',
                            'HF Hub',
                        ],
                        value="HF Hub",
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
                with gr.Row(elem_classes="gap-block-padding"):
                    ev_model_torch_dtype = gr.Dropdown(
                        label="Torch dtype",
                        choices=[
                            'auto',
                            'float16',
                            'bfloat16',
                            'float32',
                            'float64',
                            'complex64',
                            'complex128',
                        ],
                    )
                    ev_model_load_in_8bit = gr.Dropdown(
                        label="Load in 8bit",
                        choices=['Yes', 'No', 'Default'],
                        allow_custom_value=False,
                    )
                ev_model_trust_remote_code = gr.Checkbox(
                    label="Trust Remote Code"
                )
            with gr.Box(elem_classes="form-box mt-gap"):
                ev_use_custom_tokenizer = gr.Checkbox(
                    label="Use Custom Tokenizer",
                    value=False,
                )
                with gr.Column(visible=False) as ev_custom_tokenizer_settings:
                    with gr.Row(elem_classes="gap-block-padding"):
                        ev_custom_tokenizer_from = gr.Dropdown(
                            label="From",
                            choices=[
                                'Data Dir',
                                'HF Hub',
                            ],
                            value="HF Hub",
                            elem_classes="flex-grow-0"
                        )
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
            with gr.Box(elem_classes="form-box mt-gap"):
                ev_use_adapter_model = gr.Checkbox(
                    label="Use Adapter Model (LoRA)",
                    value=False,
                )
                with gr.Column(visible=False) as ev_adapter_model_settings:
                    with gr.Row(elem_classes="gap-block-padding"):
                        ev_adapter_model_from = gr.Dropdown(
                            label="From",
                            choices=[
                                'Data Dir',
                                'HF Hub',
                            ],
                            value="HF Hub",
                            elem_classes="flex-grow-0"
                        )
                        ev_adapter_model_name = gr.Textbox(
                            label="Adapter (LoRA) Model Name",
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
                ev_use_adapter_model.change(
                    fn=lambda enabled: gr.Column.update(visible=enabled),
                    inputs=[ev_use_adapter_model],
                    outputs=[ev_adapter_model_settings],  # type: ignore
                )
            gr.Markdown("### Advanced", elem_classes="mt-xxl mb-m8")
            with gr.Accordion(
                label="Advanced",
                open=False,
                elem_id="models_edit_advanced_accordion",
                elem_classes="accordion gap-0 mt-gap",
            ):
                ev_json = gr.Code(label="JSON", language='json')
                ev_json_message = gr.HTML('')
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
                        ['model', 'args', 'torch_dtype'], 'string'),
                    (ev_model_load_in_8bit,
                        ['model', 'args', 'load_in_8bit'], 'boolean'),
                    (ev_model_trust_remote_code,
                        ['model', 'args', 'trust_remote_code'], 'boolean'),
                    (ev_use_custom_tokenizer,
                        ['use_custom_tokenizer'], 'boolean'),
                    (ev_custom_tokenizer_name,
                        ['custom_tokenizer', 'name_or_path'], 'string'),
                    (ev_custom_tokenizer_name_select, [
                     'custom_tokenizer', 'name_or_path'], 'string'),
                    (ev_custom_tokenizer_from, ['custom_tokenizer', 'load_from'],
                        {'Data Dir': '"data_dir"', 'HF Hub': '"default"'}),
                    (ev_use_adapter_model,
                        ['use_adapter_model'], 'boolean'),
                    (ev_adapter_model_name,
                        ['adapter_model', 'name_or_path'], 'string'),
                    (ev_adapter_model_name_select, [
                     'adapter_model', 'name_or_path'], 'string'),
                    (ev_adapter_model_from, ['adapter_model', 'load_from'],
                        {'Data Dir': '"data_dir"', 'HF Hub': '"default"'}),
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

        things_that_might_hang.append(
            new_button.click(
                fn=handle_new_model_preset,
                inputs=[],
                outputs=[
                    main_view,  # type: ignore
                    edit_view,  # type: ignore
                    ev_title,
                    ev_uid,
                    ev_original_file_name,
                    ev_json,
                ]
            )
        )
        things_that_might_hang.append(
            edit_model_preset_btn.click(
                fn=handle_edit_model_preset,
                inputs=[model_preset_uid_to_edit],
                outputs=[
                    main_view,  # type: ignore
                    edit_view,  # type: ignore
                    ev_title,
                    ev_uid,
                    ev_original_file_name,
                    ev_json,
                ]
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
