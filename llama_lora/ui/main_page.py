import gradio as gr

from ..globals import Global
from ..models import get_model_with_lora

from .inference_ui import inference_ui

from .js_scripts import popperjs_core_code, tippy_js_code


def main_page():
    title = get_page_title()

    with gr.Blocks(
            title=title,
            css=main_page_custom_css()) as main_page_blocks:
        gr.Markdown(f"""
            <h1 class="app_title_text">{title}</h1> <wbr /><h2 class="app_subtitle_text">{Global.ui_subtitle}</h2>
            """)
        with gr.Tab("Inference"):
            inference_ui()
        if Global.ui_show_sys_info:
            gr.Markdown(f"""
                <small>Data dir: `{Global.data_dir}`</small>
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
        title = f"[UI DEV MODE] {title}"
    if (Global.ui_emoji):
        title = f"{Global.ui_emoji} {title}"
    return title


def main_page_custom_css():
    css = """
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

    #inference_preview_prompt_container .label-wrap {
        user-select: none;
    }

    #inference_preview_prompt {
        padding: 0;
    }
    #inference_preview_prompt textarea {
        border: 0;
    }

    @media screen and (min-width: 640px) {
        #inference_lora_model {
            border-top-right-radius: 0;
            border-bottom-right-radius: 0;
            border-right: 0;
            margin-right: -16px;
        }

        #inference_prompt_template {
            border-top-left-radius: 0;
            border-bottom-left-radius: 0;
            border-left: 0;

            margin-right: -90px;
            padding-right: 80px;
        }

        #inference_reload_selections_button {
            margin: 16px;
            margin-bottom: auto;
            height: 42px !important;
            min-width: 42px !important;
            width: 42px !important;
            z-index: 1;
        }
    }



    @media screen and (max-width: 392px) {
        #inference_lora_model {
            border-bottom-left-radius: 0;
            border-bottom-right-radius: 0;
        }

        #inference_prompt_template {
            border-top-left-radius: 0;
            border-top-right-radius: 0;
            border-top: 0;
            margin-top: -16px;
        }
    }

    .tippy-box[data-animation=scale-subtle][data-placement^=top]{transform-origin:bottom}.tippy-box[data-animation=scale-subtle][data-placement^=bottom]{transform-origin:top}.tippy-box[data-animation=scale-subtle][data-placement^=left]{transform-origin:right}.tippy-box[data-animation=scale-subtle][data-placement^=right]{transform-origin:left}.tippy-box[data-animation=scale-subtle][data-state=hidden]{transform:scale(.8);opacity:0}
    """
    return css
