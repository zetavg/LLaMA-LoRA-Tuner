import os
import traceback
import re
import gradio as gr
import math

from ...config import Config
from ...utils.prompter import Prompter

from .data_processing import get_data_from_input


def refresh_preview(
    template,
    load_dataset_from,
    dataset_from_data_dir,
    dataset_text,
    dataset_text_format,
    dataset_plain_text_input_variables_separator,
    dataset_plain_text_input_and_output_separator,
    dataset_plain_text_data_separator,
    max_preview_count,
):
    try:
        prompter = Prompter(template)
        variable_names = prompter.get_variable_names()

        data = get_data_from_input(
            load_dataset_from=load_dataset_from,
            dataset_text=dataset_text,
            dataset_text_format=dataset_text_format,
            dataset_plain_text_input_variables_separator=dataset_plain_text_input_variables_separator,
            dataset_plain_text_input_and_output_separator=dataset_plain_text_input_and_output_separator,
            dataset_plain_text_data_separator=dataset_plain_text_data_separator,
            dataset_from_data_dir=dataset_from_data_dir,
            prompter=prompter
        )

        train_data = prompter.get_train_data_from_dataset(
            data, max_preview_count)

        train_data = train_data[:max_preview_count]

        data_count = len(data)

        headers = ['Prompt', 'Completion']
        preview_data = [
            [item.get("prompt", ""), item.get("completion", "")]
            for item in train_data
        ]

        if not prompter.template_module:
            variable_names = prompter.get_variable_names()
            headers += [f"Variable: {variable_name}" for variable_name in variable_names]
            variables = [
                [item.get(f"_var_{name}", "") for name in variable_names]
                for item in train_data
            ]
            preview_data = [d + v for d, v in zip(preview_data, variables)]

        preview_info_message = f"The dataset has about {data_count} item(s)."
        if data_count > max_preview_count:
            preview_info_message += f" Previewing the first {max_preview_count}."

        info_message = f"about {data_count} item(s)."
        if load_dataset_from == "Data Dir":
            info_message = "This dataset contains about " + info_message
        update_message = gr.Markdown.update(info_message, visible=True)

        return (
            gr.Dataframe.update(
                value={'data': preview_data, 'headers': headers}),
            gr.Markdown.update(preview_info_message),
            update_message,
            update_message
        )
    except Exception as e:
        update_message = gr.Markdown.update(
            f"<span class=\"finetune_dataset_error_message\">Error: {e}.</span>",
            visible=True)
        return (
            gr.Dataframe.update(value={'data': [], 'headers': []}),
            gr.Markdown.update(
                "Set the dataset in the \"Prepare\" tab, then preview it here."),
            update_message,
            update_message
        )


def refresh_dataset_items_count(
    template,
    load_dataset_from,
    dataset_from_data_dir,
    dataset_text,
    dataset_text_format,
    dataset_plain_text_input_variables_separator,
    dataset_plain_text_input_and_output_separator,
    dataset_plain_text_data_separator,
    max_preview_count,
):
    try:
        prompter = Prompter(template)

        data = get_data_from_input(
            load_dataset_from=load_dataset_from,
            dataset_text=dataset_text,
            dataset_text_format=dataset_text_format,
            dataset_plain_text_input_variables_separator=dataset_plain_text_input_variables_separator,
            dataset_plain_text_input_and_output_separator=dataset_plain_text_input_and_output_separator,
            dataset_plain_text_data_separator=dataset_plain_text_data_separator,
            dataset_from_data_dir=dataset_from_data_dir,
            prompter=prompter
        )

        train_data = prompter.get_train_data_from_dataset(
            data)
        data_count = len(train_data)

        preview_info_message = f"The dataset contains {data_count} item(s)."
        if data_count > max_preview_count:
            preview_info_message += f" Previewing the first {max_preview_count}."

        info_message = f"{data_count} item(s)."
        if load_dataset_from == "Data Dir":
            info_message = "This dataset contains " + info_message
        update_message = gr.Markdown.update(info_message, visible=True)

        return (
            gr.Markdown.update(preview_info_message),
            update_message,
            update_message,
            gr.Slider.update(maximum=math.floor(data_count / 2))
        )
    except Exception as e:
        update_message = gr.Markdown.update(
            f"<span class=\"finetune_dataset_error_message\">Error: {e}.</span>",
            visible=True)

        trace = traceback.format_exc()
        traces = [s.strip() for s in re.split("\n * File ", trace)]
        traces_to_show = [s for s in traces if os.path.join(
            Config.data_dir, "templates") in s]
        traces_to_show = [re.sub(" *\n *", ": ", s) for s in traces_to_show]
        if len(traces_to_show) > 0:
            update_message = gr.Markdown.update(
                f"<span class=\"finetune_dataset_error_message\">Error: {e} ({','.join(traces_to_show)}).</span>",
                visible=True)

        return (
            gr.Markdown.update(
                "Set the dataset in the \"Prepare\" tab, then preview it here."),
            update_message,
            update_message,
            gr.Slider.update(maximum=1)
        )
