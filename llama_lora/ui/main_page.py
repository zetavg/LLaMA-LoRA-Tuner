import gradio as gr

from ..globals import Global

from .inference_ui import inference_ui
from .finetune_ui import finetune_ui
from .tokenizer_ui import tokenizer_ui

from .js_scripts import popperjs_core_code, tippy_js_code


def main_page():
    title = get_page_title()

    with gr.Blocks(
            title=title,
            css=main_page_custom_css(),
    ) as main_page_blocks:
        with gr.Column(elem_id="main_page_content"):
            gr.Markdown(f"""
                <h1 class="app_title_text">{title}</h1> <wbr />
                <h2 class="app_subtitle_text">{Global.ui_subtitle}</h2>
                """)
            with gr.Tab("Inference"):
                inference_ui()
            with gr.Tab("Fine-tuning"):
                finetune_ui()
            with gr.Tab("Tokenizer"):
                tokenizer_ui()
            info = []
            if Global.version:
                info.append(f"LLaMA-LoRA Tuner `{Global.version}`")
            info.append(f"Base model: `{Global.default_base_model_name}`")
            if Global.ui_show_sys_info:
                info.append(f"Data dir: `{Global.data_dir}`")
            gr.Markdown(f"""
                <small>{"&nbsp;&nbsp;·&nbsp;&nbsp;".join(info)}</small>
                """)
    main_page_blocks.load(_js=f"""
    function () {{
        {popperjs_core_code()}
        {tippy_js_code()}
    """ + """
        // Sync theme to body.
        setTimeout(function () {
          const gradio_container_element = document.querySelector(
            ".gradio-container"
          );
          function handle_gradio_container_element_class_change() {
            if (Array.from(gradio_container_element.classList).includes("dark")) {
              document.body.classList.add("dark");
            } else {
              document.body.classList.remove("dark");
            }
          }
          new MutationObserver(function (mutationsList, observer) {
            handle_gradio_container_element_class_change();
          }).observe(gradio_container_element, {
            attributes: true,
            attributeFilter: ["class"],
          });
          handle_gradio_container_element_class_change();
        }, 500);
    }
    """)


def get_page_title():
    title = Global.ui_title
    if (Global.ui_dev_mode):
        title = Global.ui_dev_mode_title_prefix + title
    if (Global.ui_emoji):
        title = f"{Global.ui_emoji} {title}"
    return title


def main_page_custom_css():
    css = """
    /* to make position stick work */
    .gradio-container {
        overflow-x: initial !important;
        overflow-y: clip !important;
    }

    .app_title_text {
        display: inline-block;
        margin-right: 0.5em !important;
    }
    .app_subtitle_text {
        display: inline-block;
        margin-top: 0 !important;
        font-weight: 100 !important;
        font-size: var(--text-md) !important;
    }

    .tippy-box {
        background-color: var(--block-background-fill);
        border: 1px solid var(--border-color-primary);
        border-radius: 4px;
        box-shadow: 0 2px 20px rgba(5,5,5,.08);
    }
    .tippy-arrow {
        color: var(--block-background-fill);
    }
    .tippy-content {
        color: var(--block-label-text-color);
        font-family: var(--font);
        font-weight: 100;
    }

    /*
    .codemirror-wrapper .cm-editor .cm-gutters {
        background-color: var(--background-fill-secondary);
    }
    */

    .error-message, .error-message p {
        color: var(--error-text-color) !important;
    }

    .textbox_that_is_only_used_to_display_a_label {
        border: 0 !important;
        box-shadow: none !important;
        padding: 0 !important;
    }
    .textbox_that_is_only_used_to_display_a_label textarea {
        display: none;
    }

    #main_page_content > .tabs > .tab-nav * {
        font-size: 1rem;
        font-weight: 700;
        /* text-transform: uppercase; */
    }

    #inference_lora_model_group {
        border-radius: var(--block-radius);
        background: var(--block-background-fill);
    }
    #inference_lora_model_group #inference_lora_model {
        background: transparent;
    }
    #inference_lora_model_prompt_template_message:not(.hidden) + #inference_lora_model {
        padding-bottom: 28px;
    }
    #inference_lora_model_group > #inference_lora_model_prompt_template_message {
        position: absolute;
        bottom: 8px;
        left: 20px;
        z-index: 1;
        font-size: 12px;
        opacity: 0.7;
    }
    #inference_lora_model_group > #inference_lora_model_prompt_template_message p {
        font-size: 12px;
    }
    #inference_lora_model_prompt_template_message > .wrap {
        display: none;
    }
    #inference_lora_model > .wrap:first-child:not(.hide),
    #inference_prompt_template > .wrap:first-child:not(.hide) {
        opacity: 0.5;
    }
    #inference_lora_model_group, #inference_lora_model {
        z-index: 60;
    }
    #inference_prompt_template {
        z-index: 55;
    }

    #inference_prompt_box > *:first-child {
        border-bottom-left-radius: 0;
        border-bottom-right-radius: 0;
    }
    #inference_prompt_box > *:last-child {
        margin-top: -16px;
        border-top: 0;
        border-top-left-radius: 0;
        border-top-right-radius: 0;
    }

    #inference_prompt_box {
        flex-grow: 0 !important;
    }

    #inference_options_accordion {
        padding: 0;
    }
    #inference_options_accordion > .label-wrap {
        user-select: none;
        padding: var(--block-padding);
        margin-bottom: 0;
    }
    #inference_options_accordion > *:last-child > .form {
        border-left: 0;
        border-right: 0;
        border-bottom: 0;
        border-top-left-radius: 0;
        border-top-right-radius: 0;
        box-shadow: none;
    }

    .inference_options_group {
        margin-top: -16px;
        margin-bottom: -16px;
    }
    .inference_options_group > .form {
        border-radius: 0;
        border-left: 0;
        border-right: 0;
        border-bottom: 0;
        box-shadow: none;
    }

    #inference_options_bottom_group {
        margin-top: -12px;
    }
    #inference_options_bottom_group > .form {
        border-top-left-radius: 0;
        border-top-right-radius: 0;
        border-left: 0;
        border-right: 0;
        border-bottom: 0;
    }

    #inference_output > .wrap:first-child,
    #inference_raw_output > .wrap:first-child {
        /* allow users to select text while generation is still in progress */
        pointer-events: none;

        padding: 12px !important;
    }

    /* position sticky */
    #inference_output_group_container {
        display: block;
    }
    #inference_output_group {
        position: -webkit-sticky;
        position: sticky;
        top: 16px;
        bottom: 16px;
    }

    #dataset_plain_text_input_variables_separator textarea,
    #dataset_plain_text_input_and_output_separator textarea,
    #dataset_plain_text_data_separator textarea {
        font-family: var(--font-mono);
    }
    #dataset_plain_text_input_and_output_separator,
    #dataset_plain_text_data_separator {
        margin-top: -8px;
    }

    #finetune_dataset_text_load_sample_button {
        margin: -4px 12px 8px;
    }

    #inference_preview_prompt_container .label-wrap {
        user-select: none;
    }

    #inference_preview_prompt {
        padding: 0;
    }
    #inference_preview_prompt textarea {
        border: 0;
    }
    #inference_preview_prompt > .wrap {
        pointer-events: none;
        background: transparent;
        opacity: 0.8;
    }

    #inference_update_prompt_preview_btn {
        position: absolute;
        z-index: 1;
        right: 0;
        bottom: 0;
        width: 32px;
        border-top-right-radius: 0;
        border-bottom-left-radius: 0;
        box-shadow: none;
        opacity: 0.8;
    }

    #finetune_reload_selections_button {
        position: absolute;
        top: 0;
        right: 0;
        margin: 16px;
        margin-bottom: auto;
        height: 42px !important;
        min-width: 42px !important;
        width: 42px !important;
        z-index: 1;
    }

    #finetune_dataset_from_data_dir {
        border: 0;
        box-shadow: none;
    }

    @media screen and (min-width: 640px) {
        #inference_lora_model, #inference_lora_model_group,
        #finetune_template {
            border-top-right-radius: 0;
            border-bottom-right-radius: 0;
            border-right: 0;
            margin-right: -16px;
        }
        #inference_lora_model_group #inference_lora_model {
            box-shadow: var(--block-shadow);
        }

        #inference_prompt_template {
            border-top-left-radius: 0;
            border-bottom-left-radius: 0;
            border-left: 0;

            margin-right: -90px;
            padding-right: 80px;
        }

        #finetune_template + * {
            border-top-left-radius: 0;
            border-bottom-left-radius: 0;
            border-left: 0;

            margin-right: -90px;
        }

        #finetune_template + * > * {
            padding-right: 80px;
        }

        #inference_reload_selections_button, #finetune_reload_selections_button {
            position: relative;
            margin: 16px;
            margin-bottom: auto;
            height: 42px !important;
            min-width: 42px !important;
            width: 42px !important;
            z-index: 61;
        }
    }

    #finetune_ui_content > .tabs > .tab-nav::before {
        content: "Training Dataset:";
        display: flex;
        justify-content: center;
        align-items: center;
        padding-right: 12px;
        padding-left: 8px;
    }

    #finetune_template,
    #finetune_template + * {
        border: 0;
        box-shadow: none;
    }

    #finetune_dataset_text_input_group .form {
        border: 0;
        box-shadow: none;
        padding: 0;
    }

    #finetune_dataset_text_input_textbox > .wrap:last-of-type {
        margin-top: -20px;
    }

    #finetune_dataset_plain_text_separators_group * {
        font-size: 0.8rem;
    }
    #finetune_dataset_plain_text_separators_group textarea {
        height: auto !important;
    }
    #finetune_dataset_plain_text_separators_group > .form {
        gap: 0 !important;
    }

    #finetune_dataset_from_text_message p,
    #finetune_dataset_from_text_message + * p {
        font-size: 80%;
    }
    #finetune_dataset_from_text_message,
    #finetune_dataset_from_text_message *,
    #finetune_dataset_from_text_message + *,
    #finetune_dataset_from_text_message + * * {
        display: inline;
    }


    #finetune_dataset_from_data_dir_message,
    #finetune_dataset_from_data_dir_message * {
        min-height: 0 !important;
    }
    #finetune_dataset_from_data_dir_message {
        margin: -20px 24px 0;
        font-size: 0.8rem;
    }

    #finetune_dataset_from_text_message > .wrap > *:first-child,
    #finetune_dataset_from_data_dir_message > .wrap > *:first-child {
        display: none;
    }
    #finetune_dataset_from_data_dir_message > .wrap {
        top: -18px;
    }
    #finetune_dataset_from_text_message > .wrap svg,
    #finetune_dataset_from_data_dir_message > .wrap svg {
        margin: -32px -16px;
    }

    .finetune_dataset_error_message {
        color: var(--error-text-color) !important;
    }

    #finetune_dataset_preview_info_message {
        align-items: flex-end;
        flex-direction: row;
        display: flex;
        margin-bottom: -4px;
    }

    #finetune_dataset_preview td {
        white-space: pre-wrap;
    }

    #finetune_max_seq_length {
        flex: 2;
    }

    #finetune_save_total_limit,
    #finetune_save_steps,
    #finetune_logging_steps {
        min-width: min(120px,100%) !important;
        padding-top: 4px;
    }
    #finetune_save_total_limit span,
    #finetune_save_steps span,
    #finetune_logging_steps span {
        font-size: 12px;
        margin-bottom: 5px;
    }
    #finetune_save_total_limit input,
    #finetune_save_steps input,
    #finetune_logging_steps input {
        padding: 4px 8px;
    }

    @media screen and (max-width: 392px) {
        #inference_lora_model, #finetune_template {
            border-bottom-left-radius: 0;
            border-bottom-right-radius: 0;
        }

        #inference_prompt_template, #finetune_template + * {
            border-top-left-radius: 0;
            border-top-right-radius: 0;
            border-top: 0;
            margin-top: -16px;
        }
    }

    /*
    #tokenizer_encoded_tokens_input_textbox .codemirror-wrapper,
    #tokenizer_decoded_text_input_textbox .codemirror-wrapper {
        margin-bottom: -20px;
    }
    */
    #tokenizer_encoded_tokens_input_textbox,
    #tokenizer_decoded_text_input_textbox {
        overflow: hidden !important;
    }

    /* in case if there's too many logs on the previous run and made the box too high */
    #finetune_training_status:has(.wrap:not(.hide)) {
        max-height: 160px;
        height: 160px;
    }

    .foot_stop_timeoutable_btn {
        align-self: flex-end;
        border: 0 !important;
        width: auto !important;
        background: transparent !important;
        box-shadow: none !important;
        padding: 0 !important;
        font-weight: 100 !important;
        font-size: 80% !important;
        text-decoration: underline;
        opacity: 0.3;
    }
    .foot_stop_timeoutable_btn:hover {
        opacity: 0.8;
    }
    .foot_stop_timeoutable_btn:active {
        opacity: 1;
    }

    .tippy-box[data-animation=scale-subtle][data-placement^=top]{transform-origin:bottom}.tippy-box[data-animation=scale-subtle][data-placement^=bottom]{transform-origin:top}.tippy-box[data-animation=scale-subtle][data-placement^=left]{transform-origin:right}.tippy-box[data-animation=scale-subtle][data-placement^=right]{transform-origin:left}.tippy-box[data-animation=scale-subtle][data-state=hidden]{transform:scale(.8);opacity:0}
    """
    return css
