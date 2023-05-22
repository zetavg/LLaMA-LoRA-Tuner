import pdb
from typing import Any

import gradio as gr
import time
import json
from textwrap import dedent
from json5 import loads as json_loads

from ...config import Config
from ...globals import Global
from ...models import get_tokenizer
from ...data import get_model_preset_choices, get_model_preset_from_choice
from ...lib.tokenize import tokenize
from ...utils.data_processing import comparing_lists


default_tokenizer_from_hf = "huggyllama/llama-7b"


def ui_get_tokenizer(
    tokenizer_from,
    tokenizer_from_preset_selection,
    tokenizer_from_hf_name,
):
    if tokenizer_from == 'Model Preset':
        preset = get_model_preset_from_choice(tokenizer_from_preset_selection)
        if not preset:
            raise ValueError(f"Can't find preset: '{tokenizer_from_preset_selection}'.")
        return preset.tokenizer
    elif tokenizer_from == 'HF Hub':
        return get_tokenizer(
            tokenizer_from_hf_name or default_tokenizer_from_hf)
    else:
        raise ValueError(f"Invalid tokenizer_from value: '{tokenizer_from}'.")


def handle_tokenize(text, *args):
    try:
        tokenizer = ui_get_tokenizer(*args)
        tokenize_results = tokenize(text, tokenizer)
        text = comparing_lists(
            [
                [f"{i}" for i in tokenize_results['tokens']],
                [f"{i}," if c < len(tokenize_results['ids']) - 1 else f"{i}"
                 for c, i in enumerate(tokenize_results['ids'])],
            ],
            labels=['//', ''],
            max_width=42)
        text = text.rstrip()
        text = f"[\n{text}\n]"
        return text, gr.Markdown.update('', visible=False)
    except Exception as e:
        return '', gr.Markdown.update("Error: " + str(e), visible=True)


def handle_decode(json_str, *args):
    try:
        tokenizer = ui_get_tokenizer(*args)
        ids = json_loads(json_str)
        text = tokenizer.decode(ids)
        return text, gr.Markdown.update("", visible=False)
    except Exception as e:
        return "", gr.Markdown.update("Error: " + str(e), visible=True)


def handle_encode(text):
    # base_model_name = Global.base_model_name
    tokenizer_name = Global.tokenizer_name or Global.base_model_name

    try:
        if Config.ui_dev_mode:
            return f"[\"Not actually encoding tokens in UI dev mode.\"]", gr.Markdown.update("", visible=False)
        tokenizer = get_tokenizer(tokenizer_name)
        result = tokenizer(text)
        encoded_tokens_json = json.dumps(result['input_ids'], indent=2)
        return encoded_tokens_json, gr.Markdown.update("", visible=False)
    except Exception as e:
        return "", gr.Markdown.update("Error: " + str(e), visible=True)


def tokenizer_ui():
    things_that_might_hang = []

    with gr.Blocks() as tokenizer_ui_blocks:
        with gr.Box(elem_classes="form-box"):
            with gr.Row(elem_classes=""):
                tokenizer_from = gr.Dropdown(
                    label="Use Tokenizer From",
                    value='Model Preset',
                    choices=[
                        'Model Preset',
                        'HF Hub',
                    ],
                    elem_classes="flex-grow-0")
                tokenizer_select = gr.Dropdown(
                    label="Preset",
                    visible=True
                )
                tokenizer_name = gr.Textbox(
                    label="Tokenizer Name", max_lines=1,
                    placeholder=default_tokenizer_from_hf,
                    visible=False
                )
                tokenizer_select_reload_btn = gr.Button(
                    "↻", elem_classes="block-reload-btn",
                    elem_id="tokenizer_ui_1_tokenizer_select_reload_btn")
                tokenizer_from.change(
                    fn=lambda v: (
                        gr.Dropdown.update(visible=v == 'Model Preset'),
                        gr.Textbox.update(visible=v == 'HF Hub'),
                        gr.Button.update(visible=v == 'Model Preset'),
                    ),
                    inputs=[tokenizer_from],
                    outputs=[
                        tokenizer_select,
                        tokenizer_name,
                        tokenizer_select_reload_btn,
                    ]
                )
                tokenizer_select_reload_btn.click(
                    fn=lambda v: gr.Dropdown.update(
                        value=v if v else get_model_preset_choices()[0],
                        choices=get_model_preset_choices(),
                    ),
                    inputs=[tokenizer_select],
                    outputs=[tokenizer_select]
                )
                tokenizer_selection_inputs: Any = [
                    tokenizer_from,
                    tokenizer_select,
                    tokenizer_name,
                ]
        with gr.Row():
            with gr.Column():
                tokens = gr.Code(
                    label="Tokens (JSON)",
                    language="javascript",
                    lines=10,
                    value=sample_tokens_value,
                    elem_id="tokenizer_encoded_tokens_input_textbox",
                    elem_classes="cm-max-height-400px")
                decode_btn = gr.Button("Decode ➡️")
                encoded_tokens_error_message = gr.Markdown(
                    "", visible=False, elem_classes="error-message")
            with gr.Column():
                text = gr.Code(
                    label="Text",
                    lines=10,
                    value=sample_text_value,
                    elem_id="tokenizer_decoded_text_input_textbox",
                    elem_classes="cm-max-height-400px")
                encode_btn = gr.Button("⬅️ Tokenize")
                text_error_message = gr.Markdown(
                    "", visible=False, elem_classes="error-message")

            decoding = decode_btn.click(
                fn=handle_decode,
                inputs=[tokens] + tokenizer_selection_inputs,
                outputs=[text, encoded_tokens_error_message],
            )
            tokenizing = encode_btn.click(
                fn=handle_tokenize,
                inputs=[text] + tokenizer_selection_inputs,
                outputs=[tokens, text_error_message],
            )
            things_that_might_hang.append(decoding)
            things_that_might_hang.append(tokenizing)

        stop_timeoutable_btn = gr.Button(
            "stop not-responding elements",
            elem_id="inference_stop_timeoutable_btn",
            elem_classes="foot_stop_timeoutable_btn")
        stop_timeoutable_btn.click(
            fn=None, inputs=None, outputs=None, cancels=things_that_might_hang)

    tokenizer_ui_blocks.load(_js="""
    function tokenizer_ui_blocks_js() {
      var loadingPriority = 200;

      // Load data
      setTimeout(function () {
        document.getElementById('tokenizer_ui_1_tokenizer_select_reload_btn').click();
      }, loadingPriority);

      // Tooltips
      setTimeout(function () {
        tippy('#tokenizer_ui_1_tokenizer_select_reload_btn', {
          placement: 'top',
          delay: [500, 0],
          animation: 'scale-subtle',
          content:
            'Reload selections.',
          allowHTML: true,
        });
      }, loadingPriority);
      return [];
    }
    """)


sample_tokens_value = """
[
   510, 3158,  8516,  30013, 27287, 689,
// The   quick  brown  fox    jumps  over
   253, 22658, 4370, 15
//  the  lazy   dog  .
]
"""

sample_text_value = """
The quick brown fox jumps over the lazy dog.
"""
