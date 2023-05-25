from typing import Any

import gradio as gr
import os
import re
import time
import json
from textwrap import dedent

from transformers import GenerationConfig

from ...config import Config
from ...globals import Global
# from ...models import get_model, get_tokenizer, get_device
from ...lib.csv_logger import CSVLogger
from ...data import (
    get_model_preset_choices,
    # get_available_template_names,
    get_available_lora_model_names,
    get_info_of_available_lora_model,
    get_model_preset_from_choice,
    get_prompt_samples,
    get_prompt_template_names
)
from ...utils.prompter import Prompter
from ...utils.get_time import get_time
from ...utils.relative_read_file import relative_read_file

from ..css_styles import register_css_style
from ..components.generation_options import generation_options
from ..components.model_and_prompt_template_select import model_and_prompt_template_select
from ..components.prompt_examples_select import prompt_examples_select
from ..components.markdown_output import markdown_output

from .event_handlers import (
    handle_update_conversations_list_html,
    handle_switch_conversation,
    render_current_conversation,
    clear_cached_prompters,
    pre_prepare_generate,
    prepare_generate,
    handle_generate,
    handle_stop_generate,
)
from .html_templates import (
    blank_conversation_header_content,
)

register_css_style('chat', relative_read_file(__file__, "style.css"))


def chat_ui():
    things_that_might_hang = []

    with gr.Blocks() as chat_ui_blocks:
        generation_indicator = gr.HTML(
            '',
            elem_id="chat_ui_generation_indicator"
        )
        with gr.Row():
            with gr.Column(
                scale=1,
            ):
                with gr.Column(
                    elem_classes="flex-direction-column-reverse",
                ):
                    with gr.Accordion(
                        "Raw Data", open=False,
                        elem_classes="gap-0-d3",
                    ):
                        gr.Markdown(
                            elem_classes='info-text mb-md ph-md',
                            value=dedent(f"""
                               Copy the content below to backup your conversations. Paste to restore. Or modify it directly (with tools such as [JSON Editor Online](https://jsoneditoronline.org/)).
                            """).strip()
                        )
                        sessions_str = gr.Code(
                            label="JSON",
                            elem_id="chat_ui_session_str",
                        )
                        current_conversation_id = gr.Textbox(
                            elem_id="chat_ui_current_conversation",
                            lines=1, max_lines=1,
                            visible=False,
                        )
                        current_conversation_set = gr.Button(
                            elem_id="chat_ui_current_conversation_set",
                            visible=False,
                        )
                    blank_conversations_list_container = gr.HTML(
                        value="Loading...",
                        visible=True,
                        elem_classes="chat-ui-conversations-list-blank-container",
                    )
                    with gr.Column(
                        variant="panel",
                        visible=False,
                    ) as conversations_list_container:
                        with gr.Row(
                            elem_classes="chat-ui-conversations-list-header",
                        ):
                            gr.Markdown("## Conversations")
                            new_conversation_button = gr.Button('+')
                            new_conversation_event = new_conversation_button.click(
                                fn=lambda: None,
                                outputs=current_conversation_id,
                            )
                            new_conversation_event.then(
                                fn=None,
                                _js="function () { document.getElementById('chat_ui_current_conversation_set').click(); document.getElementById('chat_ui_current_conversation_column').scrollIntoView({ behavior: 'smooth' }); document.querySelector('#chat_ui_message_input textarea').focus(); }"
                            )
                            things_that_might_hang.append(
                                new_conversation_event)
                        conversations_list_html = gr.HTML(
                            elem_classes="fix-loading-no-transform flex-grow-2",
                        )

                        with gr.Row():
                            fake_clear_conversations_btn = gr.Button(
                                "Clear",
                                elem_id="chat_ui_fake_clear_conversations_btn",
                                elem_classes="mw-fc",
                                visible=False,
                            )
                            # Not working, so we handle this in script.js.
                            fake_clear_conversations_btn.click(
                                fn=None,
                                _js=dedent('''
                                    function () {
                                        var result = confirm("Are you sure you want to clear all conversation history? This action cannot be undone.")
                                        if (result) {
                                            document.getElementById('chat_ui_clear_conversations_btn').click();
                                        }
                                        return [];
                                    }
                                    ''').strip()
                            )
                            clear_conversations_btn = gr.Button(
                                "Confirm Clear Conversations",
                                visible=False,
                                elem_id="chat_ui_clear_conversations_btn"
                            )
                            new_conversation_button_2 = gr.Button(
                                "New",
                                elem_classes="mw-fc",
                                # elem_id="chat_ui_clear_conversations_btn"
                            )
                            new_conversation_event_2 = new_conversation_button_2.click(
                                fn=lambda: None,
                                outputs=current_conversation_id,
                            )
                            new_conversation_event_2.then(
                                fn=None,
                                _js="function () { document.getElementById('chat_ui_current_conversation_set').click(); document.getElementById('chat_ui_current_conversation_column').scrollIntoView({ behavior: 'smooth' }); document.querySelector('#chat_ui_message_input textarea').focus(); }"
                            )
                            things_that_might_hang.append(
                                new_conversation_event_2)

            with gr.Column(
                scale=3,
                elem_id="chat_ui_current_conversation_column",
                elem_classes="chat-ui-current-conversation-column",
            ):
                with gr.Box(
                    elem_classes="model-and-prompt-template-select form-box disable_while_training",
                    visible=True,
                ) as model_and_prompt_template_select_box:
                    (
                        model_preset_select,
                        prompt_template_select,
                        model_prompt_template_message,
                        reload_selections_button,
                        reload_selections_event,
                        model_preset_select_change_event,
                    ) = \
                        model_and_prompt_template_select(
                            elem_id_prefix="chat_ui",
                            load_priority=200
                    )
                    reload_selections_event.then(fn=clear_cached_prompters)
                current_conversation_header_html = gr.HTML(
                    blank_conversation_header_content,
                    elem_classes="chat-ui-current-conversation-header fix-loading-no-transform",
                )
                current_conversation_chatbot_ui = gr.Chatbot(
                    visible=False,
                    show_label=False,
                    elem_classes="chat-ui-current-conversation"
                )
                with gr.Column(elem_classes="chat-ui-send-message-group gap-0-d3"):
                    with gr.Box(
                            elem_classes="send-message-box panel-with-textbox-and-btn"):
                        message = gr.Textbox(
                            label="Message",
                            placeholder="Type your message here...",
                            elem_id="chat_ui_message_input",
                            interactive=True,
                        )
                        not_used_message = gr.Textbox(
                            label="Not Used",
                            visible=False,
                        )  # TODO: So that selecting prompt_examples will not get something like `['message'']`
                        send_message_btn = gr.Button(
                            "Send",
                            variant="primary",
                            elem_id="chat_ui_send_message_btn",
                            elem_classes="send-message-btn",
                        )
                        message.submit(
                            fn=None,
                            _js="function () { document.getElementById('chat_ui_send_message_btn').click(); }"
                        )
                        stop_generation_btn = gr.Button(
                            "Stop",
                            variant="stop",
                            elem_id="chat_ui_stop_generation_btn",
                            elem_classes="stop-generation-btn",
                        )
                        stop_generation_btn.click(
                            fn=None,
                            _js=dedent('''
                                function () {
                                  if (window.chat_ui_send_message_btn_disable_timer) {
                                    clearTimeout(window.chat_ui_send_message_btn_disable_timer);
                                  }
                                  var g_btn = document.getElementById('chat_ui_send_message_btn');

                                  // To prevent double click, disable the button for 500ms.
                                  g_btn.style.pointerEvents = 'none';
                                  g_btn.style.opacity = 0.5;
                                  window.chat_ui_send_message_btn_disable_timer = setTimeout(function () {
                                    g_btn.style.pointerEvents = 'auto';
                                    g_btn.style.opacity = 1;
                                  }, 500);
                                }
                                ''').strip()
                        )
                    with gr.Accordion(
                        "Examples", open=True, visible=False,
                        elem_classes="examples-accordion accordion-with-block-title-text-color",
                    ) as prompt_examples_accordion:
                        prompt_examples_select(
                            variable_textboxes=[message, not_used_message],
                            container=prompt_examples_accordion,
                            reload_button_elem_id="chat_ui_reload_prompt_examples_button",
                            things_that_might_hang_list=things_that_might_hang,
                        )
                    with gr.Accordion(
                        "Options", open=False,
                        # elem_id="inference_options_accordion",
                        elem_classes="options-accordion gap-0-d2 accordion-with-block-title-text-color",
                    ):
                        go_component = generation_options(
                            elem_id_prefix="chat_ui",
                            elem_classes="",
                        )

                with gr.Accordion(
                    "Conversation Details",
                    # elem_id="inference_inference_tokens_output_accordion",
                    elem_classes="accordion-with-block-title-text-color mt-gap",
                    open=False,
                ):
                    with gr.Accordion(
                        "Text Output",
                        # elem_id="inference_inference_tokens_output_accordion",
                        elem_classes="accordion-with-block-title-text-color mt-md",
                        open=False,
                    ):
                        text_output = gr.Code(
                            label="Text",
                            lines=8,
                            interactive=False,
                            # elem_id="inference_tokens_output",
                            elem_classes="cm-max-height-400px")

                    with gr.Accordion(
                        "Output Tokens",
                        # elem_id="inference_inference_tokens_output_accordion",
                        elem_classes="accordion-with-block-title-text-color",
                        open=False,
                    ):
                        tokens_output = gr.Code(
                            label="JSON",
                            language="javascript",
                            lines=8,
                            interactive=False,
                            # elem_id="inference_tokens_output",
                            elem_classes="cm-max-height-400px")

        handle_update_conversations_list_html_outputs: Any = [
            blank_conversations_list_container,
            conversations_list_container,
            conversations_list_html,
            fake_clear_conversations_btn,
        ]
        render_current_conversation_inputs = [
            sessions_str, current_conversation_id
        ]
        render_current_conversation_outputs = [
            current_conversation_header_html,
            model_and_prompt_template_select_box,
            current_conversation_chatbot_ui,
            text_output,
            tokens_output,
        ]
        sessions_str.input(
            fn=handle_update_conversations_list_html,
            inputs=[sessions_str, current_conversation_id],
            outputs=handle_update_conversations_list_html_outputs,
        ).then(
            fn=None,
            inputs=[sessions_str],
            outputs=[sessions_str],
            _js="function (str) { obj = JSON.parse(str); obj._updated_at = (new Date().getTime()); return JSON.stringify(obj); }"
        )
        things_that_might_hang.append(
            current_conversation_set.click(
                fn=handle_switch_conversation,
                inputs=render_current_conversation_inputs,
                outputs=handle_update_conversations_list_html_outputs +
                render_current_conversation_outputs,  # type: ignore
            )
        )

        send_message_event = send_message_btn.click(
            fn=pre_prepare_generate,
            inputs=[
                sessions_str,
                current_conversation_id,
                model_preset_select,
                prompt_template_select,
            ],
            outputs=[
                generation_indicator,
                model_and_prompt_template_select_box,  # type: ignore
                current_conversation_header_html,
                message,
            ],
        ).then(
            fn=prepare_generate,
            inputs=[
                sessions_str,
                current_conversation_id,
                model_preset_select,
                prompt_template_select,
                message,
            ],
            outputs=(
                [
                    generation_indicator,
                    sessions_str,
                    current_conversation_id,
                    message,
                ]
                + handle_update_conversations_list_html_outputs
                + render_current_conversation_outputs)  # type: ignore
        ).then(
            fn=handle_generate,
            inputs=[
                sessions_str,
                current_conversation_id,
                go_component['generation_config_json'],
            ],
            outputs=(
                [
                    generation_indicator,
                    sessions_str,
                ]
                + render_current_conversation_outputs
            ),
        )
        # render_current_conversation_outputs = [
        #     current_conversation_header_html,
        #     model_and_prompt_template_select_box,
        #     current_conversation_chatbot_ui,
        # ]

        # .then(
        #     fn=handle_generate,
        #     inputs=[
        #         model_preset_select,
        #         prompt_template_select,
        #         go_component['generation_config_json'],
        #         stop_sequence,
        #         stream_output,
        #         variable_0, variable_1, variable_2, variable_3,
        #         variable_4, variable_5, variable_6, variable_7,
        #     ],
        #     outputs=[
        #         inference_output,
        #         inference_tokens_output,
        #         # inference_tokens_output_s,
        #         output_for_flagging,
        #     ],
        #     api_name="inference"
        # )
        things_that_might_hang.append(
            stop_generation_btn.click(
                # fn=None,
                fn=handle_stop_generate,
                inputs=[
                    sessions_str,
                    current_conversation_id,
                    # message,
                ],
                outputs=[
                    generation_indicator,
                    model_and_prompt_template_select_box,  # type: ignore
                    current_conversation_header_html,
                    conversations_list_html,
                    message,
                ],
                cancels=[send_message_event],
                # queue=False,
            )
        )

        things_that_might_hang.append(
            clear_conversations_btn.click(
                fn=lambda: (json.dumps({'_updated_at': get_time()}), None),
                inputs=[],
                outputs=[
                    sessions_str,  # type: ignore
                    current_conversation_id,
                ],
                cancels=[send_message_event],
            ).then(
                fn=handle_update_conversations_list_html,
                inputs=[sessions_str, current_conversation_id],
                outputs=handle_update_conversations_list_html_outputs,
            ).then(
                fn=handle_switch_conversation,
                inputs=render_current_conversation_inputs,
                outputs=handle_update_conversations_list_html_outputs +
                render_current_conversation_outputs,  # type: ignore
            )
        )

    stop_non_responding_elements_btn = gr.Button(
        "stop non-responding elements",
        elem_classes="foot-stop-non-responding-elements-btn")
    stop_non_responding_elements_btn.click(
        fn=None, inputs=None, outputs=None,
        cancels=things_that_might_hang)

    local_storage_key = 'llm_tuner_chat_sessions'
    load_session_btn = gr.Button(
        'Load Session from LocalStorage',
        elem_id='chat_ui_load_session_btn',
        visible=False,
    )
    load_session_2_btn = gr.Button(
        'Load Session from LocalStorage 2',
        elem_id='chat_ui_load_session_2_btn',
        visible=False,
    )
    store_session_btn = gr.Button(
        'Store Session to LocalStorage',
        elem_id='chat_ui_store_session_btn',
        visible=False,
    )
    load_session_btn.click(
        fn=None,
        _js="""
        function () {
        """ + f"""
          var local_storage_key = {json.dumps(local_storage_key)}
        """ + """
          var localStorageKey = 'llm_tuner_chat_sessions';

          try {
            var dataJson = localStorage.getItem(localStorageKey);

            // TODO: Check JSON schema

            return [JSON.stringify(JSON.parse(dataJson))];
          } catch (e) {
            console.error('Error while loading chat sessions from localStorage', e);
            return ['{ }'];
          } finally {
            setTimeout(function () {
                document.getElementById('chat_ui_load_session_2_btn').click();
            }, 10);
          }
        }
        """,
        outputs=[sessions_str],
    )
    load_session_2_btn.click(
        fn=handle_update_conversations_list_html,
        inputs=[sessions_str, current_conversation_id],
        outputs=handle_update_conversations_list_html_outputs,
    )

    store_session_btn.click(
        fn=None,
        inputs=[sessions_str],
        _js="""
        function (sessions_str) {
        """ + f"""
            if (!sessions_str) return;
            var local_storage_key = {json.dumps(local_storage_key)}
        """ + """
            var sessions = JSON.parse(sessions_str);
            if (!sessions) return;
            var updated_at = sessions['_updated_at'];
            if (updated_at && updated_at > (window.llm_tuner_chat_ui_session_stored_at || 0)) {
                localStorage.setItem(local_storage_key, sessions_str);
                window.llm_tuner_chat_ui_session_stored_at = updated_at;
              // console.log('session saved to localStorag');
            } else {
              // console.log('no updates');
            }
        }
        """,
    )
    # chat_ui_blocks.load(
    #     # fn=lambda x: [x],
    #     _js="""
    #     function (sessionData) {
    #     """ + f"""
    #         var local_storage_key = {json.dumps(local_storage_key)}
    #         debugger
    #     """ + """
    #         var updated_at = sessionData['_updated_at'];
    #         if (updated_at && updated_at > (window.llm_tuner_chat_ui_session_stored_at || 0)) {
    #             localStorage.setItem(local_storage_key, JSON.stringify(sessionData));
    #             window.llm_tuner_chat_ui_session_stored_at = updated_at;
    #           console.log('saved to localstorag');
    #         } else {
    #           console.log('no updates');
    #         }
    #     }
    #     """,
    #     inputs=[sessions],
    #     # every=1,
    # )
    chat_ui_blocks.load(_js=relative_read_file(__file__, "script.js"))
