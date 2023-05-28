from typing import Any

import gradio as gr

from ...config import Config
from ...data import (
    get_model_preset_choices,
    get_prompt_template_names,
    get_model_preset_from_choice,
)

from ..ui_utils import get_random_hex
from ..css_styles import register_css_style


def model_and_prompt_template_select(
    elem_id_prefix,
    load_priority=100,
):
    uid = get_random_hex()
    elem_id = f"model_and_prompt_template_select{uid}"
    with gr.Blocks() as model_and_prompt_select_blocks:
        with gr.Row(
            elem_id=elem_id,
            elem_classes="model-and-prompt-template-select"
        ):
            model_preset_select = gr.Dropdown(
                label="Model",
                elem_id=f"{elem_id_prefix}_model_preset_select",
            )
            prompt_template_select = gr.Dropdown(
                label="Prompt Template",
                value="None",
                elem_id=f"{elem_id_prefix}_prompt_template_select",
            )
        model_prompt_template_message = gr.HTML(
            visible=False,
            elem_classes="mt-m2 ph-2 o-09"
        )
        if Config.ui_model_and_prompt_template_select_notice:
            gr.Markdown(
                Config.ui_model_and_prompt_template_select_notice,
                elem_classes="mt-m4 ph-4 info-text"
            )
        reload_selections_button = gr.Button(
            "↻", elem_classes="block-reload-btn",
            elem_id=f"{elem_id_prefix}_model_and_prompt_template_select_reload_selections_button")

        handle_model_preset_select_change_inputs: Any = \
            [model_preset_select, prompt_template_select]
        handle_model_preset_select_change_outputs: Any = \
            [model_preset_select, prompt_template_select]

        reload_selections_event = reload_selections_button.click(
            handle_reload_selections,
            inputs=[model_preset_select, prompt_template_select],
            outputs=[model_preset_select, prompt_template_select],
            # queue=False,
        ).then(
            fn=handle_model_preset_select_change,
            inputs=handle_model_preset_select_change_inputs,
            outputs=handle_model_preset_select_change_outputs,
            # queue=False,
        )
        model_preset_select_change_event = model_preset_select.change(
            fn=handle_model_preset_select_change,
            # fn=lambda x, y: (
            #     getattr(
            #         get_model_preset_from_choice(x),
            #         'default_prompt_template',
            #         y
            #     )
            # ),
            inputs=handle_model_preset_select_change_inputs,
            outputs=handle_model_preset_select_change_outputs,
            # queue=False,
        )

    model_and_prompt_select_blocks.load(
        _js=f"""
        function () {{
            setTimeout(function () {{
                document.getElementById('{reload_selections_button.elem_id}').click();
            }}, {load_priority})
            return [];
        }}
        """
    )

    # if Config.ui_show_starter_tooltips:
    #     model_and_prompt_select_blocks.load(
    #         _js=f"""
    #         function () {{
    #             setTimeout(function () {{
    #                 add_tooltip('#{elem_id}', {{
    #                   placement: 'bottom',
    #                   content:
    #                     'Examples are loaded from the <code>prompt_samples</code> folder of your data dir.',
    #                 }});
    #             }}, 100);
    #             return [];
    #         }}
    #         """
    #     )

    return (
        model_preset_select,
        prompt_template_select,
        model_prompt_template_message,
        reload_selections_button,
        reload_selections_event,
        model_preset_select_change_event,
    )


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


def handle_model_preset_select_change(x, y):
    model_preset = get_model_preset_from_choice(x)
    if not model_preset:
        return (x, y)

    default_prompt_template = model_preset.default_prompt_template
    if not default_prompt_template or default_prompt_template == 'None':
        return (x, y)

    prompt_template_names = get_prompt_template_names()
    if default_prompt_template not in prompt_template_names:
        return (x, y)

    return (x, default_prompt_template)


register_css_style(
    'model_and_prompt_template_select_component',
    '''
    .model-and-prompt-template-select .block > .wrap.default:not(.hide) {
        opacity: 0.8;
    }
    '''
)
