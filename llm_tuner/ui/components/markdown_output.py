from typing import Any

import re
from textwrap import dedent

import gradio as gr

from ...data import get_prompt_samples

from ..css_styles import register_css_style


def markdown_output(
    elem_id,
    textbox_args={},
    markdown_container_args={},
    render_throttle=100,
    markdown_first=False,
):
    tabs = ['text', 'markdown']
    if markdown_first:
        tabs = ['markdown', 'text']

    with gr.Blocks() as blocks:
        with gr.Column(
            elem_id=elem_id,
            elem_classes=f"markdown-output-group{' markdown-first' if markdown_first else ''}",
        ):
            for t in tabs:
                if t == 'text':
                    with gr.Tab(
                        'Text',
                        elem_classes="markdown-output-group-text"
                    ):
                        text_output = gr.Textbox(
                            **{
                                'elem_id': f"{elem_id}_textbox",
                                **textbox_args,
                            }
                        )
                elif t == 'markdown':
                    with gr.Tab(
                        'Markdown',
                        elem_classes="markdown-output-group-markdown hljs-github-theme"
                    ):
                        markdown_output = gr.HTML(
                            **{
                                'elem_id': f"{elem_id}_markdown",
                                **markdown_container_args,
                            }
                        )

    set_variables_js_code = dedent(f"""
        var textarea = document.querySelector('#{text_output.elem_id} textarea');
        var md_content_elem = document.querySelector('#{markdown_output.elem_id} .prose');
        var render_throttle = {render_throttle};
    """).strip()

    blocks.load(
        _js=dedent("""
        function () {
            // Markdown rendering
            setTimeout(function () {
        """ + set_variables_js_code + """
              function throttle(func, limit) {
                var lastRan;
                var timer;
                return function() {
                  const context = this;
                  const args = arguments;
                  function runFunc() {
                    lastRan = Date.now();
                    func.apply(context, args);
                  }

                  if (timer) clearTimeout(timer);
                  if (!lastRan || (Date.now() - lastRan) > limit) {
                    runFunc();
                  } else {
                    timer = setTimeout(runFunc, limit - (Date.now() - lastRan));
                  }
                }
              }

              var throttled_render = throttle(function () {
                var html = window.markdownit_md.render(textarea.value);
                md_content_elem.innerHTML = html;
              }, render_throttle);

              throttled_render();

              textarea.addEventListener('input', throttled_render);
              textarea.addEventListener('change', throttled_render);

              var observer = new MutationObserver(function (mutations) {
                throttled_render();
              });

              observer.observe(textarea, { attributes: true, childList: true, characterData: true, subtree: true });
            }, 100);

            return [];
        }
        """).strip()
    )

    return (text_output, markdown_output)


register_css_style(
    'markdown_output_component',
    '''
    .markdown-output-group .markdown-output-group-text {
        display: block !important;
    }
    .markdown-output-group .markdown-output-group-text[style="display: none;"] {
        padding-bottom: 0;
    }
    .markdown-output-group .markdown-output-group-text[style="display: none;"] textarea {
        display: none;
    }

    .markdown-output-group.markdown-first > .tabs {
        flex-direction: column-reverse;
    }

    .markdown-output-group {
        box-shadow: var(--block-shadow);
        border: var(--block-border-width) solid var(--border-color-primary);
        border-radius: var(--block-radius);
        background: var(--block-background-fill);
    }

    .markdown-output-group .tab-nav {
        border: 0;
        position: absolute;
        z-index: 1;
        top: 0;
        right: 14px;
        padding: var(--block-padding);
        gap: var(--spacing-md);
        opacity: 0.8;
    }
    .markdown-output-group .tab-nav button {
        margin-top: -4px;
        border: 0;
        border-radius: 100px;
        height: 25px;
        padding: 2px 12px;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    .markdown-output-group .tabitem {
        border: 0;
    }

    .markdown-output-group .markdown-output-group-text .form,
    .markdown-output-group .markdown-output-group-text .block {
        border: 0 !important;
        padding: 0 !important;
    }

    .markdown-output-group .markdown-output-group-markdown .prose {
        min-height: 42px;
        box-shadow: var(--input-shadow);
        border: var(--input-border-width) solid var(--input-border-color);
        border-radius: var(--input-radius);
        background: var(--input-background-fill);
        padding: var(--input-padding);
        max-height: 500px;
        overflow: auto;
    }

    .markdown-output-group .markdown-output-group-markdown {
        padding-top: 0 !important;
    }

    .markdown-output-group .markdown-output-group-markdown .prose:empty::after {
        content: 'Empty';
        opacity: 0.2;
        text-transform: uppercase;
        font-weight: 800;
    }

    .markdown-output-group .markdown-output-group-markdown .prose:empty {
        display: flex;
        justify-content: center;
        align-items: center;
    }

    .markdown-output-group .markdown-output-group-text .form, .markdown-output-group .markdown-output-group-text > div {
        overflow: visible !important;
    }
    /*
    .markdown-output-group .markdown-output-group-text button.copy-text {
        top: -10px;
        right: -12px;
    }
    */

    .markdown-output-group .markdown-output-group-text,
    .markdown-output-group .markdown-output-group-text > *,
    .markdown-output-group .markdown-output-group-text .form,
    .markdown-output-group .markdown-output-group-text .form > * {
        position: static !important;
    }

    .markdown-output-group .wrap.default {
        pointer-events: none;
    }
    '''
)
