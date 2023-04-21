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
            with gr.Row():
                gr.Markdown(
                    f"""
                    <h1 class="app_title_text">{title}</h1> <wbr />
                    <h2 class="app_subtitle_text">{Global.ui_subtitle}</h2>
                    """,
                    elem_id="page_title",
                )
                with gr.Column(elem_id="global_base_model_select_group"):
                    global_base_model_select = gr.Dropdown(
                        label="Base Model",
                        elem_id="global_base_model_select",
                        choices=Global.base_model_choices,
                        value=lambda: Global.base_model_name,
                        allow_custom_value=True,
                    )
                    use_custom_tokenizer_btn = gr.Button(
                        "Use custom tokenizer",
                        elem_id="use_custom_tokenizer_btn")
                    global_tokenizer_select = gr.Dropdown(
                        label="Tokenizer",
                        elem_id="global_tokenizer_select",
                        # choices=[],
                        value=lambda: Global.base_model_name,
                        visible=False,
                        allow_custom_value=True,
                    )
                    use_custom_tokenizer_btn.click(
                        fn=lambda: gr.Dropdown.update(visible=True),
                        inputs=None,
                        outputs=[global_tokenizer_select])
            # global_base_model_select_loading_status = gr.Markdown("", elem_id="global_base_model_select_loading_status")

            with gr.Column(elem_id="main_page_tabs_container") as main_page_tabs_container:
                with gr.Tab("Inference"):
                    inference_ui()
                with gr.Tab("Fine-tuning"):
                    finetune_ui()
                with gr.Tab("Tokenizer"):
                    tokenizer_ui()
            please_select_a_base_model_message = gr.Markdown(
                "Please select a base model.", visible=False)
            current_base_model_hint = gr.Markdown(
                lambda: Global.base_model_name, elem_id="current_base_model_hint")
            current_tokenizer_hint = gr.Markdown(
                lambda: Global.tokenizer_name, elem_id="current_tokenizer_hint")
            foot_info = gr.Markdown(get_foot_info)

    global_base_model_select.change(
        fn=pre_handle_change_base_model,
        inputs=[global_base_model_select],
        outputs=[main_page_tabs_container]
    ).then(
        fn=handle_change_base_model,
        inputs=[global_base_model_select],
        outputs=[
            main_page_tabs_container,
            please_select_a_base_model_message,
            current_base_model_hint,
            current_tokenizer_hint,
            # global_base_model_select_loading_status,
            foot_info
        ]
    )

    global_tokenizer_select.change(
        fn=pre_handle_change_tokenizer,
        inputs=[global_tokenizer_select],
        outputs=[main_page_tabs_container]
    ).then(
        fn=handle_change_tokenizer,
        inputs=[global_tokenizer_select],
        outputs=[
            global_tokenizer_select,
            main_page_tabs_container,
            current_tokenizer_hint,
            foot_info
        ]
    )

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
    """ + """
        setTimeout(function () {
          // Workaround default value not shown.
          const current_base_model_hint_elem = document.querySelector('#current_base_model_hint > p');
          if (!current_base_model_hint_elem) return;

          const base_model_name = current_base_model_hint_elem.innerText;
          document.querySelector('#global_base_model_select input').value = base_model_name;
          document.querySelector('#global_base_model_select').classList.add('show');

          const current_tokenizer_hint_elem = document.querySelector('#current_tokenizer_hint > p');
          const tokenizer_name = current_tokenizer_hint_elem && current_tokenizer_hint_elem.innerText;

          if (tokenizer_name && tokenizer_name !== base_model_name) {
            const btn = document.getElementById('use_custom_tokenizer_btn');
            if (btn) btn.click();
          }
        }, 3200);
    """ + """
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
        /* box-shadow: var(--shadow-drop-lg); */
    }
    body.dark .tippy-box {
        box-shadow: 0 0 8px rgba(160,160,160,0.12);
    }
    .tippy-arrow {
        color: var(--block-background-fill);
    }
    .tippy-content {
        color: var(--block-label-text-color);
        font-family: var(--font);
        font-weight: 100;
    }

    .tippy-arrow::before {
        z-index: 1;
    }
    .tippy-arrow::after {
        content: "";
        position: absolute;
        z-index: -1;
        border-color: transparent;
        border-style: solid;
    }
    .tippy-box[data-placement^=top]> .tippy-arrow::after {
        bottom: -9px;
        left: -1px;
        border-width: 9px 9px 0;
        border-top-color: var(--border-color-primary);
        transform-origin: center top;
    }
    .tippy-box[data-placement^=bottom]> .tippy-arrow::after {
        top: -9px;
        left: -1px;
        border-width: 0 9px 9px;
        border-bottom-color: var(--border-color-primary);
        transform-origin: center bottom;
    }
    .tippy-box[data-placement^=left]> .tippy-arrow:after {
        border-width: 9px 0 9px 9px;
        border-left-color: var(--border-color-primary);
        top: -1px;
        right: -9px;
        transform-origin: center left;
    }
    .tippy-box[data-placement^=right]> .tippy-arrow::after {
        top: -1px;
        left: -9px;
        border-width: 9px 9px 9px 0;
        border-right-color: var(--border-color-primary);
        transform-origin: center right;
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

    .flex_vertical_grow_area {
        margin-top: calc(var(--layout-gap) * -1) !important;
        flex-grow: 1 !important;
        max-height: calc(var(--layout-gap) * 2);
    }
    .flex_vertical_grow_area.no_limit {
        max-height: unset;
    }

    #page_title {
        flex-grow: 3;
    }
    #global_base_model_select_group,
    #global_base_model_select,
    #global_tokenizer_select {
        position: relative;
        align-self: center;
        min-width: 250px !important;
    }
    #global_base_model_select,
    #global_tokenizer_select {
        position: relative;
        padding: 2px 2px;
        border: 0;
        box-shadow: none;
    }
    #global_base_model_select {
        opacity: 0;
        pointer-events: none;
    }
    #global_base_model_select.show {
        opacity: 1;
        pointer-events: auto;
    }
    #global_base_model_select label .wrap-inner,
    #global_tokenizer_select label .wrap-inner {
        padding: 2px 8px;
    }
    #global_base_model_select label span,
    #global_tokenizer_select label span {
        margin-bottom: 2px;
        font-size: 80%;
        position: absolute;
        top: -14px;
        left: 8px;
        opacity: 0;
    }
    #global_base_model_select_group:hover label span,
    #global_base_model_select:hover label span,
    #global_tokenizer_select:hover label span {
        opacity: 1;
    }
    #use_custom_tokenizer_btn {
        position: absolute;
        top: -16px;
        right: 10px;
        border: 0 !important;
        width: auto !important;
        background: transparent !important;
        box-shadow: none !important;
        padding: 0 !important;
        font-weight: 100 !important;
        text-decoration: underline;
        font-size: 12px !important;
        opacity: 0;
    }
    #global_base_model_select_group:hover #use_custom_tokenizer_btn {
        opacity: 0.3;
    }

    #global_base_model_select_loading_status {
        position: absolute;
        pointer-events: none;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
    }
    #global_base_model_select_loading_status > .wrap:not(.hide) {
        z-index: 9999;
        position: absolute;
        top: 112px !important;
        bottom: 0 !important;
        max-height: none;
        background: var(--background-fill-primary);
        opacity: 0.8;
    }
    #global_base_model_select ul {
        z-index: 9999;
        background: var(--block-background-fill);
    }

    #current_base_model_hint, #current_tokenizer_hint {
        display: none;
    }

    #main_page_content > .tabs > .tab-nav * {
        font-size: 1rem;
        font-weight: 700;
        /* text-transform: uppercase; */
    }

    #inference_reload_selected_models_btn {
        position: absolute;
        top: 0;
        left: 0;
        width: 0;
        height: 0;
        padding: 0;
        opacity: 0;
        pointer-events: none;
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
    #inference_lora_model_group {
        flex-direction: column-reverse;
        border-width: var(--block-border-width);
        border-color: var(--block-border-color);
    }
    #inference_lora_model_group #inference_lora_model {
        border: 0;
    }
    #inference_lora_model_group > #inference_lora_model_prompt_template_message {
        padding: var(--block-padding) !important;
        padding-bottom: 5px !important;
        margin-top: -50px !important;
        margin-left: 4px !important;
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

    #inference_flagging_group {
        position: relative;
        margin-top: -8px;
        margin-bottom: -8px;
        gap: calc(var(--layout-gap) / 2);
    }
    #inference_flag_output {
        min-height: 1px !important;
        position: absolute;
        top: 0;
        bottom: 0;
        right: 0;
        pointer-events: none;
        opacity: 0.5;
    }
    #inference_flag_output .wrap {
        top: 0;
        bottom: 0;
        right: 0;
        justify-content: center;
        align-items: flex-end;
        padding: 4px !important;
    }
    #inference_flag_output .wrap svg {
        display: none;
    }
    .form:has(> #inference_output_for_flagging),
    #inference_output_for_flagging {
        display: none;
    }
    #inference_flagging_group:has(#inference_output_for_flagging.hidden) {
        opacity: 0.5;
        pointer-events: none;
    }
    #inference_flag_up_btn, #inference_flag_down_btn {
        min-width: 44px;
        flex-grow: 1;
    }
    #inference_flag_btn {
        flex-grow: 2;
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

    #finetune_continue_from_model_box {
        /* padding: 0; */
    }
    #finetune_continue_from_model_box .block {
        border: 0;
        box-shadow: none;
        padding: 0;
    }
    #finetune_continue_from_model_box > * {
        /* gap: 0; */
    }
    #finetune_continue_from_model_box button {
        margin-top: 16px;
    }
    #finetune_continue_from_model {
        flex-grow: 2;
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

    /*
    #finetune_dataset_preview {
        max-height: 100vh;
        overflow: auto;
        border: var(--block-border-width) solid var(--border-color-primary);
        border-radius: var(--radius-lg);
    }
    #finetune_dataset_preview .table-wrap {
        border: 0 !important;
    }
    */

    #finetune_max_seq_length {
        flex: 2;
    }

    #finetune_lora_target_modules_box,
    #finetune_lora_target_modules_box + #finetune_lora_modules_to_save_box {
        margin-top: calc((var(--layout-gap) + 8px) * -1);
        flex-grow: 0 !important;
    }
    #finetune_lora_target_modules_box > .form,
    #finetune_lora_target_modules_box + #finetune_lora_modules_to_save_box > .form {
        padding-top: calc((var(--layout-gap) + 8px) / 3);
        border-top: 0;
        border-top-left-radius: 0;
        border-top-right-radius: 0;
        background: var(--block-background-fill);
        position: relative;
    }
    #finetune_lora_target_modules_box > .form::before,
    #finetune_lora_target_modules_box + #finetune_lora_modules_to_save_box > .form::before {
        content: "";
        display: block;
        position: absolute;
        top: calc((var(--layout-gap) + 8px) / 3);
        left: 0;
        right: 0;
        height: 1px;
        z-index: 1;
        background: var(--block-border-color);
    }
    #finetune_lora_target_modules_add_box,
    #finetune_lora_modules_to_save_add_box {
        margin-top: -24px;
        padding-top: 8px;
        border-top-left-radius: 0;
        border-top-right-radius: 0;
        border-top: 0;
    }
    #finetune_lora_target_modules_add_box > * > .form,
    #finetune_lora_modules_to_save_add_box > * > .form {
        border: 0;
        box-shadow: none;
    }
    #finetune_lora_target_modules_add,
    #finetune_lora_modules_to_save_add {
        padding: 0;
    }
    #finetune_lora_target_modules_add input,
    #finetune_lora_modules_to_save_add input {
        padding: 4px 8px;
    }
    #finetune_lora_target_modules_add_btn,
    #finetune_lora_modules_to_save_add_btn {
        min-width: 60px;
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

    #finetune_advanced_options_checkboxes > * > * {
        min-width: auto;
    }

    #finetune_log_and_save_options_group_container {
        flex-grow: 0 !important;
    }
    #finetune_model_name_group {
        flex-grow: 0 !important;
    }

    #finetune_eval_data_group {
        flex-grow: 0 !important;
    }

    #finetune_additional_training_arguments_box > .form,
    #finetune_additional_lora_config_box > .form {
        border: 0;
        background: transparent;
    }
    #finetune_additional_training_arguments_textbox_for_label_display,
    #finetune_additional_lora_config_textbox_for_label_display {
        padding: 0;
        margin-bottom: -10px;
        background: transparent;
    }
    #finetune_additional_training_arguments_textbox_for_label_display textarea,
    #finetune_additional_lora_config_textbox_for_label_display textarea {
        display: none;
    }

    @media screen and (max-width: 392px) {
        #inference_lora_model, #inference_lora_model_group, #finetune_template {
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


def pre_handle_change_base_model(selected_base_model_name):
    if Global.base_model_name != selected_base_model_name:
        return gr.Column.update(visible=False)
    if Global.tokenizer_name and Global.tokenizer_name != selected_base_model_name:
        return gr.Column.update(visible=False)
    return gr.Column.update(visible=True)


def handle_change_base_model(selected_base_model_name):
    Global.base_model_name = selected_base_model_name
    Global.tokenizer_name = selected_base_model_name

    is_base_model_selected = False
    if Global.base_model_name:
        is_base_model_selected = True

    return (
        gr.Column.update(visible=is_base_model_selected),
        gr.Markdown.update(visible=not is_base_model_selected),
        Global.base_model_name,
        Global.tokenizer_name,
        get_foot_info())


def pre_handle_change_tokenizer(selected_tokenizer_name):
    if Global.tokenizer_name != selected_tokenizer_name:
        return gr.Column.update(visible=False)
    return gr.Column.update(visible=True)


def handle_change_tokenizer(selected_tokenizer_name):
    Global.tokenizer_name = selected_tokenizer_name

    show_tokenizer_select = True
    if not Global.tokenizer_name:
        show_tokenizer_select = False
    if Global.tokenizer_name == Global.base_model_name:
        show_tokenizer_select = False

    return (
        gr.Dropdown.update(visible=show_tokenizer_select),
        gr.Column.update(visible=True),
        Global.tokenizer_name,
        get_foot_info()
    )


def get_foot_info():
    info = []
    if Global.version:
        info.append(f"LLaMA-LoRA Tuner `{Global.version}`")
    if Global.base_model_name:
        info.append(f"Base model: `{Global.base_model_name}`")
    if Global.tokenizer_name and Global.tokenizer_name != Global.base_model_name:
        info.append(f"Tokenizer: `{Global.tokenizer_name}`")
    if Global.ui_show_sys_info:
        info.append(f"Data dir: `{Global.data_dir}`")
    return f"""\
        <small>{"&nbsp;&nbsp;·&nbsp;&nbsp;".join(info)}</small>
        """
