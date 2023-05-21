import gradio as gr

from ...utils.relative_read_file import relative_read_file
from ..css_styles import register_css_style

from ..ui_utils import mk_fake_btn, tie_controls_with_json_editor

from .event_handlers import (
    handle_load_model_presets,
    handle_show_model_preset,
    handle_new_model_preset,
    handle_edit_model_preset,
    handle_delete_model_preset,
    handle_save_edit,
    handle_discard_edit,
)

register_css_style('models', relative_read_file(__file__, "style.css"))


def models_ui():
    things_that_might_hang = []
    with gr.Blocks() as models_ui_blocks:
        gr.Markdown("This is models_ui", elem_id="models_ui")

        with gr.Column() as main_view:
            gr.Markdown("""
                ## Model Presets
                Model presets are a way to save and load model configurations.
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
                        things_that_might_hang.append(
                            reload_button.click(
                                fn=handle_load_model_presets,
                                inputs=[],
                                outputs=[model_presets_list]
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

                with gr.Column(scale=3,
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

            things_that_might_hang.append(
                show_model_preset_btn.click(
                    fn=handle_show_model_preset,
                    inputs=[model_preset_uid_to_show],
                    outputs=[
                        right_column_empty,  # type: ignore
                        right_column_show,  # type: ignore
                        rcs_title,
                        rcs_json,
                    ]
                )
            )

        with gr.Box(visible=False) as edit_view:
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
                ev_model_name = gr.Textbox(
                    label="Model Name",
                    placeholder="Example: huggyllama/llama-7b",
                    lines=1, max_lines=1)
                with gr.Row(elem_classes="gap-0"):
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
                ev_use_adapter_model = gr.Checkbox(
                    label="Use Adapter Model (LoRA)",
                    value=False,
                )
                with gr.Column(visible=False) as ev_adapter_model_settings:
                    ev_adapter_model_name = gr.Textbox(
                        label="Adapter (LoRA) Model Name",
                        lines=1, max_lines=1)
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
                    (ev_model_torch_dtype,
                        ['model', 'args', 'torch_dtype'], 'string'),
                    (ev_model_load_in_8bit,
                        ['model', 'args', 'load_in_8bit'], 'boolean'),
                    (ev_model_trust_remote_code,
                        ['model', 'args', 'trust_remote_code'], 'boolean'),
                    (ev_use_adapter_model,
                        ['use_adapter_model'], 'boolean'),
                    (ev_adapter_model_name,
                        ['adapter_model', 'name_or_path'], 'string'),
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
                outputs=[
                    model_presets_list,
                    main_view,  # type: ignore
                    right_column_empty,   # type: ignore
                    edit_view,  # type: ignore
                    right_column_show,   # type: ignore
                ]
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
                outputs=[model_presets_list],
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

    models_ui_blocks.load(_js=relative_read_file(__file__, "script.js"))
