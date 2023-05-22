import gradio as gr

from ..config import Config
from ..globals import Global

from .inference.inference_ui import inference_ui
from .finetune.finetune_ui import finetune_ui
from .models.models_ui import models_ui
from .tools import tools_ui

from .js_scripts import popperjs_core_code, tippy_js_code
from .css_styles import get_css_styles, register_css_style

from . import styles as styles


def main_page():
    title = get_page_title()

    with gr.Blocks(
            title=title,
            css=get_css_styles(),
    ) as main_page_blocks:
        training_indicator = gr.HTML(
            "", visible=False, elem_id="training_indicator")
        with gr.Column(elem_id="main_page_content"):
            with gr.Row():
                gr.Markdown(
                    f"""
                    <h1 class="app_title_text">{title}</h1> <wbr />
                    <h2 class="app_subtitle_text">{Config.ui_subtitle}</h2>
                    """,
                    elem_id="page_title",
                )

            with gr.Column(
                elem_id="main_page_tabs_container"
            ) as main_page_tabs_container:
                with gr.Tab("Inference"):
                    inference_ui()
                with gr.Tab("Models"):
                    models_ui()
                with gr.Tab("Fine-tuning"):
                    finetune_ui()
                with gr.Tab("Tools"):
                    tools_ui()

            foot_info = gr.Markdown(get_foot_info)

    main_page_blocks.load(
        fn=lambda: gr.HTML.update(
            visible=Global.is_training or Global.is_train_starting,
            value=Global.is_training and "training"
            or (
                Global.is_train_starting and "train_starting" or ""
            )
        ),
        inputs=None,
        outputs=[training_indicator],
        every=3
    )

    main_page_blocks.load(_js=f"""
    function () {{
        {popperjs_core_code()}
        {tippy_js_code()}
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
      return [];
    }
    """)


def get_page_title():
    title = Config.ui_title
    if (Config.ui_dev_mode):
        title = Config.ui_dev_mode_title_prefix + title
    if (Config.ui_emoji):
        title = f"{Config.ui_emoji} {title}"
    return title


def main_page_custom_css():
    css = """
    /* to make position sticky work */
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

    /*
    .codemirror-wrapper .cm-editor .cm-gutters {
        background-color: var(--background-fill-secondary);
    }
    */

   .hide_wrap > .wrap {
       border: 0;
       background: transparent;
       pointer-events: none;
   }

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

    #training_indicator { display: none; }
    #training_indicator:not(.hidden) ~ * .disable_while_training {
        position: relative !important;
        pointer-events: none !important;
    }
    #training_indicator:not(.hidden) ~ * .disable_while_training * {
        pointer-events: none !important;
    }
    #training_indicator:not(.hidden) ~ * .disable_while_training::after {
        content: "Disabled while training is in progress";
        display: flex;
        position: absolute !important;
        z-index: 70;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: var(--block-background-fill);
        opacity: 0.7;
        justify-content: center;
        align-items: center;
        color: var(--body-text-color);
        font-size: var(--text-lg);
        font-weight: var(--weight-bold);
        text-transform: uppercase;
    }
    #training_indicator:not(.hidden) ~ * .disable_while_training.without_message::after {
        content: "";
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

    /*
    #dataset_plain_text_input_variables_separator textarea,
    #dataset_plain_text_input_and_output_separator textarea,
    #dataset_plain_text_data_separator textarea {
        font-family: var(--font-mono);
    }
    #dataset_plain_text_input_and_output_separator,
    #dataset_plain_text_data_separator {
        margin-top: -8px;
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
    */

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
    """
    return css


register_css_style('main', main_page_custom_css())


def get_foot_info():
    info = []
    if Global.version:
        info.append(f"LLaMA-LoRA Tuner `{Global.version}`")
    if Config.ui_show_sys_info:
        info.append(f"Data dir: `{Config.data_dir}`")
    return f"""\
        <small>{"&nbsp;&nbsp;·&nbsp;&nbsp;".join(info)}</small>
        """
