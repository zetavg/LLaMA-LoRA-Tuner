from typing import Any

import re
from textwrap import dedent

import gradio as gr

from ...data import get_prompt_samples

from ..css_styles import register_css_style


def prompt_examples_select(
    variable_textboxes,
    container,
    reload_button_elem_id,
    things_that_might_hang_list=None,
):
    with gr.Blocks() as blocks:
        with gr.Row(
            elem_classes="prompt-examples-select gap-block-padding"
        ):
            prompt_examples = gr.State({})
            prompt_examples_category_select = gr.Dropdown(
                # elem_classes="example_category_select",
                elem_classes="category-select",
                choices=[],
                show_label=False,
                interactive=True,
            )
            prompt_examples_select = gr.Dropdown(
                # elem_classes="example_select",
                elem_classes="select",
                value=None,
                choices=[],
                show_label=False,
                interactive=True,
            )

        reload_prompt_examples_button = gr.Button(
            "↻", elem_classes="block-reload-btn",
            elem_id=reload_button_elem_id)

        def handle_reload_prompt_examples(selected_category):
            prompt_examples = get_prompt_samples()

            accordion_updates = {}
            if prompt_examples:
                accordion_updates['visible'] = True

            if selected_category not in prompt_examples.keys():
                selected_category = next(
                    iter(prompt_examples), None)

            example_choices = []
            if selected_category:
                example_choices = [
                    "{:2d}. ".format(i + 1) + next(iter(c), '')
                    for i, c in enumerate(prompt_examples[selected_category])]

            return (
                prompt_examples,
                gr.Accordion.update(**accordion_updates),
                gr.Dropdown.update(
                    value=selected_category,
                    choices=list(prompt_examples.keys()),
                ),
                gr.Dropdown.update(
                    choices=example_choices,
                )
            )

        reload_prompt_examples_event = reload_prompt_examples_button.click(
            fn=handle_reload_prompt_examples,
            inputs=[
                prompt_examples_category_select
            ],
            outputs=[
                prompt_examples,
                container,  # type: ignore
                prompt_examples_category_select,
                prompt_examples_select
            ]
        )
        if isinstance(things_that_might_hang_list, list):
            things_that_might_hang_list.append(reload_prompt_examples_event)

        def handle_example_category_select(name, samples):
            example_choices = []
            if name in samples:
                example_choices = [
                    "{:2d}. ".format(i + 1) + next(iter(c), '')
                    for i, c in enumerate(samples[name])]

            return gr.Dropdown.update(
                choices=example_choices,
            )

        def handle_example_choose(c, i, samples):
            return_value = []

            if c in samples and len(samples[c]) > i:
                sample = samples[c][i]
                return_value = sample

            while len(return_value) < len(variable_textboxes):
                return_value.append(gr.Textbox.update())
            while len(return_value) > len(variable_textboxes):
                return_value.pop()

            return return_value

        def handle_example_select(c, s, samples):
            i = 0
            m = re.match(f'^ *([0-9]+)', s)
            if m:
                i = int(m.group(1)) - 1
            return handle_example_choose(c, i, samples)

        example_category_select_event = prompt_examples_category_select.select(
            fn=handle_example_category_select,
            inputs=[
                prompt_examples_category_select,
                prompt_examples],
            outputs=[prompt_examples_select]
        )
        if isinstance(things_that_might_hang_list, list):
            things_that_might_hang_list.append(example_category_select_event)

        prompt_examples_select.select(
            fn=handle_example_select,
            inputs=[
                prompt_examples_category_select,
                prompt_examples_select,
                prompt_examples],
            outputs=variable_textboxes
        ).then(
            fn=None,
            inputs=[],
            outputs=[prompt_examples_select],
            _js=dedent(f"""
                function () {{
                    setTimeout(function () {{
                        document.querySelector(
                            '#{variable_textboxes[0].elem_id} textarea'
                        ).focus();
                    }}, 200);
                    return [null];
                }}
                """).strip()
        )

    blocks.load(
        _js=dedent("""
        function () {
            // Add placeholder
            setTimeout(function () {
              document.querySelectorAll('.prompt-examples-select .select input').forEach(function (elem) { elem.placeholder = 'Select an example...' });
            }, 50);

            // Auto load examples
            setTimeout(function () {
        """ + f"""
                document.getElementById('{reload_button_elem_id}').click();
        """ + """
              }, 150);
            return [];
        }
        """).strip()
    )


register_css_style(
    'prompt_examples_select_component',
    '''
    .prompt-examples-select .block {
        border: 0 !important;
        padding: 0;
        box-shadow: none;
    }
    .prompt-examples-select .category-select .wrap-inner,
    .prompt-examples-select .select .wrap-inner {
        padding: var(--spacing-xxs) var(--spacing-sm) !important;
    }
    .prompt-examples-select .category-select .wrap-inner .dropdown-arrow,
    .prompt-examples-select .select .wrap-inner .dropdown-arrow {
        margin-right: var(--size-1) !important;
    }
    #inference_prompt_box .examples .wrap-inner input {
        margin-right: 0 !important;
    }
    .prompt-examples-select .category-select {
        flex-grow: 0;
        flex-shrink: 1;
        min-width: min(100px, 100%);
    }
    .prompt-examples-select input::placeholder {
        color: var(--input-placeholder-color);
    }
    .prompt-examples-select .category-select input {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .prompt-examples-select .select {
        flex-shrink: 1;
    }
    .prompt-examples-select .select .options .item {
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        padding: 0;
        border: var(--size-2) solid transparent;
        text-indent: -20px;
        white-space: pre-wrap;
    }
    .prompt-examples-select .select .options .inner-item {
        display: inline-block !important;
        width: 20px !important;
        height: 20px !important;
        float: left !important;
    }
    '''
)
