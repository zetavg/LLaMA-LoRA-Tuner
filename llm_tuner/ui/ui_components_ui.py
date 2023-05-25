import pdb
from typing import Any

import gradio as gr
import time
import json
from textwrap import dedent
from json5 import loads as json_loads

from .components.markdown_output import markdown_output
from .components.generation_options import generation_options
from .components.model_and_prompt_template_select import model_and_prompt_template_select
from .components.prompt_examples_select import prompt_examples_select


def ui_components_ui():
    things_that_might_hang = []

    with gr.Blocks() as ui_components_ui_blocks:
        gr.Markdown("""
            # UI Components

            Reusable UI components.
        """)

        gr.Markdown("## Markdown output")
        with gr.Column():
            markdown_output_textbox, markdown_output_html = \
                markdown_output(
                    "ui_demo_markdown_output",
                    {
                        'label': "Output",
                        'lines': 12,
                        'interactive': True,
                        'value': sample_markdown,
                    },
                    {
                        'label': "Output",
                    },
                    markdown_first=True,
                )
            markdown_output_textbox.style(show_copy_button=True)
            gr.Markdown("### Actions")
            markdown_output_set_to_example_btn = gr.Button(
                "Set to example",
                elem_classes="width-auto align-self-flex-start"
            )
            markdown_output_set_to_example_btn.style(size='sm')
            markdown_output_set_to_example_btn.click(
                fn=None,
                inputs=[],
                outputs=[markdown_output_textbox],
                _js=f"function () {{ return {json.dumps(sample_markdown, ensure_ascii=False)} }}"
            )

        gr.HTML('<hr />', elem_classes="mt-md mb-xxl")

        gr.Markdown("## Generation Options")
        with gr.Column():
            generation_options()

        gr.HTML('<hr />', elem_classes="mt-md mb-xxl")

        gr.Markdown("## Model and Prompt Template Select")
        with gr.Column():
            with gr.Box(elem_classes="form-box disable_while_training"):
                model_and_prompt_template_select(elem_id_prefix="ui_demo")

        gr.HTML('<hr />', elem_classes="mt-md mb-xxl")

        gr.Markdown("## Prompt Examples Select")
        with gr.Column():
            variable_textboxes = [
                gr.Textbox(
                    label="Instruction",
                    elem_id="ui_demo_prompt_examples_first_textbox"
                ),
                gr.Textbox(
                    label="Input"
                ),
            ]
            with gr.Box() as prompt_examples_select_container:
                prompt_examples_select(
                    variable_textboxes=variable_textboxes,
                    container=prompt_examples_select_container,
                    reload_button_elem_id="ui_demo_prompt_examples_select_reload_button",
                )

    ui_components_ui_blocks.load(_js="""
    function ui_components_ui_blocks_js() {
      return [];
    }
    """)


sample_markdown = r"""
# This is a header

This is some regular text. With some **bold**, _italic_, ~~trikethrough~~, and `code` text. And a [link](https://www.example.com).

Here is a codeblock:

```python
def hello_world():
    print("Hello, world!")
```

And here is a LaTeX equation:

$$
\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}
$$

This is a sentence with some inline LaTeX, like $E = mc^2$.

## This is another header

Here is a list:

- item 1
- item 2
- item 3
""".strip()
