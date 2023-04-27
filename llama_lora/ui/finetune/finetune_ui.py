import os
import json
from datetime import datetime
import gradio as gr
from random_word import RandomWords

from ...config import Config
from ...globals import Global
from ...utils.data import (
    get_available_template_names,
    get_available_dataset_names,
    get_available_lora_model_names
)
from ...utils.relative_read_file import relative_read_file
from ..css_styles import register_css_style

from .values import (
    default_dataset_plain_text_input_variables_separator,
    default_dataset_plain_text_input_and_output_separator,
    default_dataset_plain_text_data_separator,
    sample_plain_text_value,
    sample_jsonl_text_value,
    sample_json_text_value,
)
from .previewing import (
    refresh_preview,
    refresh_dataset_items_count,
)
from .training import (
    do_train,
    render_training_status,
    render_loss_plot
)

register_css_style('finetune', relative_read_file(__file__, "style.css"))


def random_hyphenated_word():
    r = RandomWords()
    word1 = r.get_random_word()
    word2 = r.get_random_word()
    return word1 + '-' + word2


def random_name():
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")
    return f"{random_hyphenated_word()}-{formatted_datetime}"


def reload_selections(current_template, current_dataset):
    available_template_names = get_available_template_names()
    available_template_names_with_none = available_template_names + ["None"]
    if current_template not in available_template_names_with_none:
        current_template = None
    current_template = current_template or next(
        iter(available_template_names_with_none), None)

    available_dataset_names = get_available_dataset_names()
    if current_dataset not in available_dataset_names:
        current_dataset = None
    current_dataset = current_dataset or next(
        iter(available_dataset_names), None)

    available_lora_models = ["-"] + get_available_lora_model_names()

    return (
        gr.Dropdown.update(
            choices=available_template_names_with_none,
            value=current_template),
        gr.Dropdown.update(
            choices=available_dataset_names,
            value=current_dataset),
        gr.Dropdown.update(choices=available_lora_models)
    )


def handle_switch_dataset_source(source):
    if source == "Text Input":
        return gr.Column.update(visible=True), gr.Column.update(visible=False)
    else:
        return gr.Column.update(visible=False), gr.Column.update(visible=True)


def handle_switch_dataset_text_format(format):
    if format == "Plain Text":
        return gr.Column.update(visible=True)
    return gr.Column.update(visible=False)


def load_sample_dataset_to_text_input(format):
    if format == "JSON":
        return gr.Code.update(value=sample_json_text_value)
    if format == "JSON Lines":
        return gr.Code.update(value=sample_jsonl_text_value)
    else:  # Plain Text
        return gr.Code.update(value=sample_plain_text_value)


def handle_continue_from_model_change(model_name):
    try:
        lora_models_directory_path = os.path.join(
            Config.data_dir, "lora_models")
        lora_model_directory_path = os.path.join(
            lora_models_directory_path, model_name)
        all_files = os.listdir(lora_model_directory_path)
        checkpoints = [
            file for file in all_files if file.startswith("checkpoint-")]
        checkpoints = ["-"] + checkpoints
        can_load_params = "finetune_params.json" in all_files or "finetune_args.json" in all_files
        return (gr.Dropdown.update(choices=checkpoints, value="-"),
                gr.Button.update(visible=can_load_params),
                gr.Markdown.update(value="", visible=False))
    except Exception:
        pass
    return (gr.Dropdown.update(choices=["-"], value="-"),
            gr.Button.update(visible=False),
            gr.Markdown.update(value="", visible=False))


def handle_load_params_from_model(
    model_name,
    template, load_dataset_from, dataset_from_data_dir,
    max_seq_length,
    evaluate_data_count,
    micro_batch_size,
    gradient_accumulation_steps,
    epochs,
    learning_rate,
    train_on_inputs,
    lora_r,
    lora_alpha,
    lora_dropout,
    lora_target_modules,
    lora_modules_to_save,
    load_in_8bit,
    fp16,
    bf16,
    gradient_checkpointing,
    save_steps,
    save_total_limit,
    logging_steps,
    additional_training_arguments,
    additional_lora_config,
    lora_target_module_choices,
    lora_modules_to_save_choices,
):
    error_message = ""
    notice_message = ""
    unknown_keys = []
    try:
        lora_models_directory_path = os.path.join(
            Config.data_dir, "lora_models")
        lora_model_directory_path = os.path.join(
            lora_models_directory_path, model_name)

        try:
            with open(os.path.join(lora_model_directory_path, "info.json"), "r") as f:
                info = json.load(f)
                if isinstance(info, dict):
                    model_prompt_template = info.get("prompt_template")
                    if model_prompt_template:
                        template = model_prompt_template
                    model_dataset_name = info.get("dataset_name")
                    if model_dataset_name and isinstance(model_dataset_name, str) and not model_dataset_name.startswith("N/A"):
                        load_dataset_from = "Data Dir"
                        dataset_from_data_dir = model_dataset_name
        except FileNotFoundError:
            pass

        data = {}
        possible_files = ["finetune_params.json", "finetune_args.json"]
        for file in possible_files:
            try:
                with open(os.path.join(lora_model_directory_path, file), "r") as f:
                    data = json.load(f)
            except FileNotFoundError:
                pass

        for key, value in data.items():
            if key == "max_seq_length":
                max_seq_length = value
            if key == "cutoff_len":
                max_seq_length = value
            elif key == "evaluate_data_count":
                evaluate_data_count = value
            elif key == "val_set_size":
                evaluate_data_count = value
            elif key == "micro_batch_size":
                micro_batch_size = value
            elif key == "gradient_accumulation_steps":
                gradient_accumulation_steps = value
            elif key == "epochs":
                epochs = value
            elif key == "num_train_epochs":
                epochs = value
            elif key == "learning_rate":
                learning_rate = value
            elif key == "train_on_inputs":
                train_on_inputs = value
            elif key == "lora_r":
                lora_r = value
            elif key == "lora_alpha":
                lora_alpha = value
            elif key == "lora_dropout":
                lora_dropout = value
            elif key == "lora_target_modules":
                lora_target_modules = value
                if value:
                    for element in value:
                        if element not in lora_target_module_choices:
                            lora_target_module_choices.append(element)
            elif key == "lora_modules_to_save":
                lora_modules_to_save = value
                if value:
                    for element in value:
                        if element not in lora_modules_to_save_choices:
                            lora_modules_to_save_choices.append(element)
            elif key == "load_in_8bit":
                load_in_8bit = value
            elif key == "fp16":
                fp16 = value
            elif key == "bf16":
                bf16 = value
            elif key == "gradient_checkpointing":
                gradient_checkpointing = value
            elif key == "save_steps":
                save_steps = value
            elif key == "save_total_limit":
                save_total_limit = value
            elif key == "logging_steps":
                logging_steps = value
            elif key == "additional_training_arguments":
                if value:
                    additional_training_arguments = json.dumps(value, indent=2)
                else:
                    additional_training_arguments = ""
            elif key == "additional_lora_config":
                if value:
                    additional_lora_config = json.dumps(value, indent=2)
                else:
                    additional_lora_config = ""
            elif key == "group_by_length":
                pass
            elif key == "resume_from_checkpoint":
                pass
            else:
                unknown_keys.append(key)
    except Exception as e:
        error_message = str(e)

    if len(unknown_keys) > 0:
        notice_message = f"Note: cannot restore unknown arg: {', '.join([f'`{x}`' for x in unknown_keys])}"

    message = ". ".join([x for x in [error_message, notice_message] if x])

    has_message = False
    if message:
        message += "."
        has_message = True

    return (
        gr.Markdown.update(value=message, visible=has_message),
        template, load_dataset_from, dataset_from_data_dir,
        max_seq_length,
        evaluate_data_count,
        micro_batch_size,
        gradient_accumulation_steps,
        epochs,
        learning_rate,
        train_on_inputs,
        lora_r,
        lora_alpha,
        lora_dropout,
        gr.CheckboxGroup.update(value=lora_target_modules,
                                choices=lora_target_module_choices),
        gr.CheckboxGroup.update(
            value=lora_modules_to_save, choices=lora_modules_to_save_choices),
        load_in_8bit,
        fp16,
        bf16,
        gradient_checkpointing,
        save_steps,
        save_total_limit,
        logging_steps,
        additional_training_arguments,
        additional_lora_config,
        lora_target_module_choices,
        lora_modules_to_save_choices
    )


default_lora_target_module_choices = ["q_proj", "k_proj", "v_proj", "o_proj"]
default_lora_modules_to_save_choices = ["model.embed_tokens", "lm_head"]


def handle_lora_target_modules_add(choices, new_module, selected_modules):
    choices.append(new_module)
    selected_modules.append(new_module)

    return (choices, "", gr.CheckboxGroup.update(value=selected_modules, choices=choices))


def handle_lora_modules_to_save_add(choices, new_module, selected_modules):
    choices.append(new_module)
    selected_modules.append(new_module)

    return (choices, "", gr.CheckboxGroup.update(value=selected_modules, choices=choices))


def do_abort_training():
    Global.should_stop_training = True
    Global.training_status_text = "Aborting..."


def finetune_ui():
    things_that_might_timeout = []

    with gr.Blocks() as finetune_ui_blocks:
        with gr.Column(elem_id="finetune_ui_content"):
            with gr.Tab("Prepare"):
                with gr.Box(elem_id="finetune_ui_select_dataset_source"):
                    with gr.Row():
                        template = gr.Dropdown(
                            label="Template",
                            elem_id="finetune_template",
                        )
                        load_dataset_from = gr.Radio(
                            ["Text Input", "Data Dir"],
                            label="Load Dataset From",
                            value="Text Input",
                            elem_id="finetune_load_dataset_from")
                        reload_selections_button = gr.Button(
                            "↻",
                            elem_id="finetune_reload_selections_button"
                        )
                        reload_selections_button.style(
                            full_width=False,
                            size="sm")
                    with gr.Column(
                        elem_id="finetune_dataset_from_data_dir_group",
                        visible=False
                    ) as dataset_from_data_dir_group:
                        dataset_from_data_dir = gr.Dropdown(
                            label="Dataset",
                            elem_id="finetune_dataset_from_data_dir",
                        )
                        dataset_from_data_dir_message = gr.Markdown(
                            "",
                            visible=False,
                            elem_id="finetune_dataset_from_data_dir_message")
                with gr.Box(elem_id="finetune_dataset_text_input_group") as dataset_text_input_group:
                    gr.Textbox(
                        label="Training Data", elem_classes="textbox_that_is_only_used_to_display_a_label")
                    dataset_text = gr.Code(
                        show_label=False,
                        language="json",
                        value=sample_plain_text_value,
                        # max_lines=40,
                        elem_id="finetune_dataset_text_input_textbox")
                    dataset_from_text_message = gr.Markdown(
                        "",
                        visible=False,
                        elem_id="finetune_dataset_from_text_message")
                    gr.Markdown(
                        "The data you entered here will not be saved. Do not make edits here directly. Instead, edit the data elsewhere then paste it here.")
                    with gr.Row():
                        with gr.Column():
                            dataset_text_format = gr.Radio(
                                ["Plain Text", "JSON Lines", "JSON"],
                                label="Format", value="Plain Text", elem_id="finetune_dataset_text_format")
                            dataset_text_load_sample_button = gr.Button(
                                "Load Sample", elem_id="finetune_dataset_text_load_sample_button")
                            dataset_text_load_sample_button.style(
                                full_width=False,
                                size="sm")
                        with gr.Column(elem_id="finetune_dataset_plain_text_separators_group") as dataset_plain_text_separators_group:
                            dataset_plain_text_input_variables_separator = gr.Textbox(
                                label="Input Variables Separator",
                                elem_id="dataset_plain_text_input_variables_separator",
                                placeholder=default_dataset_plain_text_input_variables_separator,
                                value=default_dataset_plain_text_input_variables_separator)
                            dataset_plain_text_input_and_output_separator = gr.Textbox(
                                label="Input and Output Separator",
                                elem_id="dataset_plain_text_input_and_output_separator",
                                placeholder=default_dataset_plain_text_input_and_output_separator,
                                value=default_dataset_plain_text_input_and_output_separator)
                            dataset_plain_text_data_separator = gr.Textbox(
                                label="Data Separator",
                                elem_id="dataset_plain_text_data_separator",
                                placeholder=default_dataset_plain_text_data_separator,
                                value=default_dataset_plain_text_data_separator)
                        things_that_might_timeout.append(
                            dataset_text_format.change(
                                fn=handle_switch_dataset_text_format,
                                inputs=[dataset_text_format],
                                outputs=[
                                    dataset_plain_text_separators_group  # type: ignore
                                ]
                            ))

                    things_that_might_timeout.append(
                        dataset_text_load_sample_button.click(fn=load_sample_dataset_to_text_input, inputs=[
                            dataset_text_format], outputs=[dataset_text]))
                gr.Markdown(
                    "💡 Switch to the \"Preview\" tab to verify that your inputs are correct.")
            with gr.Tab("Preview"):
                with gr.Row():
                    finetune_dataset_preview_info_message = gr.Markdown(
                        "Set the dataset in the \"Prepare\" tab, then preview it here.",
                        elem_id="finetune_dataset_preview_info_message"
                    )
                    finetune_dataset_preview_count = gr.Number(
                        label="Preview items count",
                        value=10,
                        # minimum=1,
                        # maximum=100,
                        precision=0,
                        elem_id="finetune_dataset_preview_count"
                    )
                finetune_dataset_preview = gr.Dataframe(
                    wrap=True, elem_id="finetune_dataset_preview")
            things_that_might_timeout.append(
                load_dataset_from.change(
                    fn=handle_switch_dataset_source,
                    inputs=[load_dataset_from],
                    outputs=[
                        dataset_text_input_group,
                        dataset_from_data_dir_group
                    ]  # type: ignore
                ))

            dataset_inputs = [
                template,
                load_dataset_from,
                dataset_from_data_dir,
                dataset_text,
                dataset_text_format,
                dataset_plain_text_input_variables_separator,
                dataset_plain_text_input_and_output_separator,
                dataset_plain_text_data_separator,
            ]
            dataset_preview_inputs = dataset_inputs + \
                [finetune_dataset_preview_count]

            with gr.Row():
                max_seq_length = gr.Slider(
                    minimum=1, maximum=4096, value=512,
                    label="Max Sequence Length",
                    info="The maximum length of each sample text sequence. Sequences longer than this will be truncated.",
                    elem_id="finetune_max_seq_length"
                )

                train_on_inputs = gr.Checkbox(
                    label="Train on Inputs",
                    value=True,
                    info="If not enabled, inputs will be masked out in loss.",
                    elem_id="finetune_train_on_inputs"
                )

        with gr.Row():
            # https://huggingface.co/docs/transformers/main/main_classes/trainer

            micro_batch_size_default_value = 1

            if Global.gpu_total_cores is not None and Global.gpu_total_memory is not None:
                memory_per_core = Global.gpu_total_memory / Global.gpu_total_cores
                if memory_per_core >= 6291456:
                    micro_batch_size_default_value = 8
                elif memory_per_core >= 4000000:  # ?
                    micro_batch_size_default_value = 4

            with gr.Column():
                micro_batch_size = gr.Slider(
                    minimum=1, maximum=100, step=1, value=micro_batch_size_default_value,
                    label="Micro Batch Size",
                    info="The number of examples in each mini-batch for gradient computation. A smaller micro_batch_size reduces memory usage but may increase training time."
                )

                gradient_accumulation_steps = gr.Slider(
                    minimum=1, maximum=10, step=1, value=1,
                    label="Gradient Accumulation Steps",
                    info="The number of steps to accumulate gradients before updating model parameters. This can be used to simulate a larger effective batch size without increasing memory usage."
                )

                epochs = gr.Slider(
                    minimum=1, maximum=100, step=1, value=10,
                    label="Epochs",
                    info="The number of times to iterate over the entire training dataset. A larger number of epochs may improve model performance but also increase the risk of overfitting.")

                learning_rate = gr.Slider(
                    minimum=0.00001, maximum=0.01, value=3e-4,
                    label="Learning Rate",
                    info="The initial learning rate for the optimizer. A higher learning rate may speed up convergence but also cause instability or divergence. A lower learning rate may require more steps to reach optimal performance but also avoid overshooting or oscillating around local minima."
                )

                with gr.Column(elem_id="finetune_eval_data_group"):
                    evaluate_data_count = gr.Slider(
                        minimum=0, maximum=1, step=1, value=0,
                        label="Evaluation Data Count",
                        info="The number of data to be used for evaluation. This specific amount of data will be randomly chosen from the training dataset for evaluating the model's performance during the process, without contributing to the actual training.",
                        elem_id="finetune_evaluate_data_count"
                    )
                gr.HTML(elem_classes="flex_vertical_grow_area")

                with gr.Accordion("Advanced Options", open=False, elem_id="finetune_advance_options_accordion"):
                    with gr.Row(elem_id="finetune_advanced_options_checkboxes"):
                        load_in_8bit = gr.Checkbox(
                            label="8bit", value=Config.load_8bit)
                        fp16 = gr.Checkbox(label="FP16", value=True)
                        bf16 = gr.Checkbox(label="BF16", value=False)
                        gradient_checkpointing = gr.Checkbox(
                            label="gradient_checkpointing", value=False)
                    with gr.Column(variant="panel", elem_id="finetune_additional_training_arguments_box"):
                        gr.Textbox(
                            label="Additional Training Arguments",
                            info="Additional training arguments to be passed to the Trainer. Note that this can override ALL other arguments set elsewhere. See https://bit.ly/hf20-transformers-training-arguments for more details.",
                            elem_id="finetune_additional_training_arguments_textbox_for_label_display"
                        )
                        additional_training_arguments = gr.Code(
                            label="JSON",
                            language="json",
                            value="",
                            lines=2,
                            elem_id="finetune_additional_training_arguments")

                with gr.Box(elem_id="finetune_continue_from_model_box"):
                    with gr.Row():
                        continue_from_model = gr.Dropdown(
                            value="-",
                            label="Continue from Model",
                            choices=["-"],
                            allow_custom_value=True,
                            elem_id="finetune_continue_from_model"
                        )
                        continue_from_checkpoint = gr.Dropdown(
                            value="-",
                            label="Resume from Checkpoint",
                            choices=["-"],
                            elem_id="finetune_continue_from_checkpoint")
                    with gr.Column():
                        load_params_from_model_btn = gr.Button(
                            "Load training parameters from selected model", visible=False)
                        load_params_from_model_btn.style(
                            full_width=False,
                            size="sm")
                        load_params_from_model_message = gr.Markdown(
                            "", visible=False)

                    things_that_might_timeout.append(
                        continue_from_model.change(
                            fn=handle_continue_from_model_change,
                            inputs=[continue_from_model],
                            outputs=[
                                continue_from_checkpoint,
                                load_params_from_model_btn,
                                load_params_from_model_message
                            ]
                        )
                    )

            with gr.Column():
                lora_r = gr.Slider(
                    minimum=1, maximum=16, step=1, value=8,
                    label="LoRA R",
                    info="The rank parameter for LoRA, which controls the dimensionality of the rank decomposition matrices. A larger lora_r increases the expressiveness and flexibility of LoRA but also increases the number of trainable parameters and memory usage."
                )

                lora_alpha = gr.Slider(
                    minimum=1, maximum=128, step=1, value=16,
                    label="LoRA Alpha",
                    info="The scaling parameter for LoRA, which controls how much LoRA affects the original pre-trained model weights. A larger lora_alpha amplifies the impact of LoRA but may also distort or override the pre-trained knowledge."
                )

                lora_dropout = gr.Slider(
                    minimum=0, maximum=1, value=0.05,
                    label="LoRA Dropout",
                    info="The dropout probability for LoRA, which controls the fraction of LoRA parameters that are set to zero during training. A larger lora_dropout increases the regularization effect of LoRA but also increases the risk of underfitting."
                )

                with gr.Column(elem_id="finetune_lora_target_modules_box"):
                    lora_target_modules = gr.CheckboxGroup(
                        label="LoRA Target Modules",
                        choices=default_lora_target_module_choices,
                        value=["q_proj", "v_proj"],
                        info="Modules to replace with LoRA.",
                        elem_id="finetune_lora_target_modules"
                    )
                    lora_target_module_choices = gr.State(
                        value=default_lora_target_module_choices)  # type: ignore
                    with gr.Box(elem_id="finetune_lora_target_modules_add_box"):
                        with gr.Row():
                            lora_target_modules_add = gr.Textbox(
                                lines=1, max_lines=1, show_label=False,
                                elem_id="finetune_lora_target_modules_add"
                            )
                            lora_target_modules_add_btn = gr.Button(
                                "Add",
                                elem_id="finetune_lora_target_modules_add_btn"
                            )
                            lora_target_modules_add_btn.style(
                                full_width=False, size="sm")
                    things_that_might_timeout.append(lora_target_modules_add_btn.click(
                        handle_lora_target_modules_add,
                        inputs=[lora_target_module_choices,
                                lora_target_modules_add, lora_target_modules],
                        outputs=[lora_target_module_choices,
                                 lora_target_modules_add, lora_target_modules],
                    ))

                with gr.Accordion("Advanced LoRA Options", open=False, elem_id="finetune_advance_lora_options_accordion"):
                    with gr.Column(elem_id="finetune_lora_modules_to_save_box"):
                        lora_modules_to_save = gr.CheckboxGroup(
                            label="LoRA Modules To Save",
                            choices=default_lora_modules_to_save_choices,
                            value=[],
                            # info="",
                            elem_id="finetune_lora_modules_to_save"
                        )
                        lora_modules_to_save_choices = gr.State(
                            value=default_lora_modules_to_save_choices)  # type: ignore
                        with gr.Box(elem_id="finetune_lora_modules_to_save_add_box"):
                            with gr.Row():
                                lora_modules_to_save_add = gr.Textbox(
                                    lines=1, max_lines=1, show_label=False,
                                    elem_id="finetune_lora_modules_to_save_add"
                                )
                                lora_modules_to_save_add_btn = gr.Button(
                                    "Add",
                                    elem_id="finetune_lora_modules_to_save_add_btn"
                                )
                                lora_modules_to_save_add_btn.style(
                                    full_width=False, size="sm")
                        things_that_might_timeout.append(lora_modules_to_save_add_btn.click(
                            handle_lora_modules_to_save_add,
                            inputs=[lora_modules_to_save_choices,
                                    lora_modules_to_save_add, lora_modules_to_save],
                            outputs=[lora_modules_to_save_choices,
                                     lora_modules_to_save_add, lora_modules_to_save],
                        ))

                    with gr.Column(variant="panel", elem_id="finetune_additional_lora_config_box"):
                        gr.Textbox(
                            label="Additional LoRA Config",
                            info="Additional LoraConfig. Note that this can override ALL other arguments set elsewhere.",
                            elem_id="finetune_additional_lora_config_textbox_for_label_display"
                        )
                        additional_lora_config = gr.Code(
                            label="JSON",
                            language="json",
                            value="",
                            lines=2,
                            elem_id="finetune_additional_lora_config")

                gr.HTML(elem_classes="flex_vertical_grow_area no_limit")

                with gr.Column(elem_id="finetune_log_and_save_options_group_container"):
                    with gr.Row(elem_id="finetune_log_and_save_options_group"):
                        logging_steps = gr.Number(
                            label="Logging Steps",
                            precision=0,
                            value=10,
                            elem_id="finetune_logging_steps"
                        )
                        save_steps = gr.Number(
                            label="Steps Per Save",
                            precision=0,
                            value=500,
                            elem_id="finetune_save_steps"
                        )
                        save_total_limit = gr.Number(
                            label="Saved Checkpoints Limit",
                            precision=0,
                            value=5,
                            elem_id="finetune_save_total_limit"
                        )

                with gr.Column(elem_id="finetune_model_name_group"):
                    model_name = gr.Textbox(
                        lines=1, label="LoRA Model Name", value=random_name,
                        max_lines=1,
                        info="The name of the new LoRA model.",
                        elem_id="finetune_model_name",
                    )

        with gr.Row():
            with gr.Column():
                pass
            with gr.Column():

                with gr.Row():
                    train_btn = gr.Button(
                        "Train", variant="primary", label="Train",
                        elem_id="finetune_start_btn"
                    )

                    abort_button = gr.Button(
                        "Abort", label="Abort",
                        elem_id="finetune_stop_btn"
                    )
                    confirm_abort_button = gr.Button(
                        "Confirm Abort", label="Confirm Abort", variant="stop",
                        elem_id="finetune_confirm_stop_btn"
                    )

        things_that_might_timeout.append(reload_selections_button.click(
            reload_selections,
            inputs=[template, dataset_from_data_dir],
            outputs=[template, dataset_from_data_dir, continue_from_model],
        ))

        for i in dataset_preview_inputs:
            things_that_might_timeout.append(
                i.change(
                    fn=refresh_preview,
                    inputs=dataset_preview_inputs,
                    outputs=[
                        finetune_dataset_preview,
                        finetune_dataset_preview_info_message,
                        dataset_from_text_message,
                        dataset_from_data_dir_message
                    ]
                ).then(
                    fn=refresh_dataset_items_count,
                    inputs=dataset_preview_inputs,
                    outputs=[
                        finetune_dataset_preview_info_message,
                        dataset_from_text_message,
                        dataset_from_data_dir_message,
                        evaluate_data_count,
                    ]
                ))

        finetune_args = [
            max_seq_length,
            evaluate_data_count,
            micro_batch_size,
            gradient_accumulation_steps,
            epochs,
            learning_rate,
            train_on_inputs,
            lora_r,
            lora_alpha,
            lora_dropout,
            lora_target_modules,
            lora_modules_to_save,
            load_in_8bit,
            fp16,
            bf16,
            gradient_checkpointing,
            save_steps,
            save_total_limit,
            logging_steps,
            additional_training_arguments,
            additional_lora_config,
        ]

        things_that_might_timeout.append(
            load_params_from_model_btn.click(
                fn=handle_load_params_from_model,
                inputs=(
                    [continue_from_model] +
                    [template, load_dataset_from, dataset_from_data_dir] +
                    finetune_args +
                    [lora_target_module_choices, lora_modules_to_save_choices]
                ),  # type: ignore
                outputs=(
                    [load_params_from_model_message] +
                    [template, load_dataset_from, dataset_from_data_dir] +
                    finetune_args +
                    [lora_target_module_choices, lora_modules_to_save_choices]
                )  # type: ignore
            )
        )

        train_status = gr.HTML(
            "",
            label="Train Output",
            elem_id="finetune_training_status")

        with gr.Column(visible=False, elem_id="finetune_loss_plot_container") as loss_plot_container:
            loss_plot = gr.Plot(
                visible=False, show_label=False,
                elem_id="finetune_loss_plot")

        training_indicator = gr.HTML(
            "training_indicator", visible=False, elem_id="finetune_training_indicator")

        train_start = train_btn.click(
            fn=do_train,
            inputs=(dataset_inputs + finetune_args + [
                model_name,
                continue_from_model,
                continue_from_checkpoint,
            ]),
            outputs=[train_status, training_indicator,
                     loss_plot_container, loss_plot]
        )

        # controlled by JS, shows the confirm_abort_button
        abort_button.click(None, None, None, None)
        confirm_abort_button.click(
            fn=do_abort_training,
            inputs=None, outputs=None,
            cancels=[train_start])

    training_status_updates = finetune_ui_blocks.load(
        fn=render_training_status,
        inputs=None,
        outputs=[train_status, training_indicator],
        every=0.2
    )
    loss_plot_updates = finetune_ui_blocks.load(
        fn=render_loss_plot,
        inputs=None,
        outputs=[loss_plot_container, loss_plot],
        every=10
    )
    finetune_ui_blocks.load(_js=relative_read_file(__file__, "script.js"))

    # things_that_might_timeout.append(training_status_updates)
    stop_timeoutable_btn = gr.Button(
        "stop not-responding elements",
        elem_id="inference_stop_timeoutable_btn",
        elem_classes="foot_stop_timeoutable_btn")
    stop_timeoutable_btn.click(
        fn=None, inputs=None, outputs=None, cancels=things_that_might_timeout)
