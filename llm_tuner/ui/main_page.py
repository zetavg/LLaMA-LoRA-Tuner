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
                    elem_id="global_base_model_select_group",
                    elem_classes="disable_while_training without_message"
                ):
                    global_base_model_select = gr.Dropdown(
                        label="Base Model",
                        elem_id="global_base_model_select",
                        choices=Config.base_model_choices,
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
                with gr.Tab("Models"):
                    models_ui()
                with gr.Tab("Tools"):
                    tools_ui()
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

    .tippy-box[data-animation=scale-subtle][data-placement^=top]{transform-origin:bottom}.tippy-box[data-animation=scale-subtle][data-placement^=bottom]{transform-origin:top}.tippy-box[data-animation=scale-subtle][data-placement^=left]{transform-origin:right}.tippy-box[data-animation=scale-subtle][data-placement^=right]{transform-origin:left}.tippy-box[data-animation=scale-subtle][data-state=hidden]{transform:scale(.8);opacity:0}
    """
    return css


register_css_style('main', main_page_custom_css())


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
    if Config.ui_show_sys_info:
        info.append(f"Data dir: `{Config.data_dir}`")
    return f"""\
        <small>{"&nbsp;&nbsp;·&nbsp;&nbsp;".join(info)}</small>
        """
