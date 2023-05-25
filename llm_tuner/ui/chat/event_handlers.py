from typing import Any

import os
import time
import json
import json5
import hashlib

import gradio as gr
from transformers import GenerationConfig


from ...config import Config
from ...globals import Global
from ...data import (
    get_model_preset_choices,
    get_prompt_template_names,
    get_available_lora_model_names,
    get_info_of_available_lora_model,
    get_model_preset_from_choice,
)

from ...utils.get_time import get_time
from ...utils.prompter import Prompter, remove_common_from_start
from ...utils.data_processing import comparing_lists
from ...utils.prepare_generation_config_args import prepare_generation_config_args

from .html_templates import (
    blank_conversation_header_content,
    conversation_list_html,
    conversation_header_html,
)

cached_prompters = {}


def get_prompter(name: str) -> Prompter:
    global cached_prompters
    # print('cached_prompters', cached_prompters)
    p = cached_prompters.get(name)
    if p:
        return p

    p = Prompter(name)
    cached_prompters[name] = p
    return p


def clear_cached_prompters():
    global cached_prompters
    cached_prompters = {}
    # print('cached_prompters', cached_prompters)


def handle_update_conversations_list_html(
    sessions_str, current_conversation_id
):
    sessions = {}
    try:
        sessions = json.loads(sessions_str) or {}
    except json.decoder.JSONDecodeError as e:
        print('WARN: JSONDecodeError on Chat UI sessions:', e)

    sessions_list = [
        {**v, '_id': k, '_active': k == current_conversation_id}
        for k, v in sessions.items() if not k.startswith('_')]

    container_updates = {'visible': False}
    blank_container_updates = {
        'value': 'No conversations yet',
        'visible': True,
    }
    clear_conversation_button_updates = {'visible': False}
    if sessions_list:
        container_updates['visible'] = True
        blank_container_updates['visible'] = False  # type: ignore
        clear_conversation_button_updates['visible'] = True

    return (
        gr.HTML.update(**blank_container_updates),
        gr.Column.update(**container_updates),
        gr.HTML.update(
            value=conversation_list_html(sessions_list),
        ),
        gr.Button.update(**clear_conversation_button_updates),
    )


def handle_switch_conversation(
    sessions_str, current_conversation_id,
    **kwargs
):
    conversations_list_output = handle_update_conversations_list_html(
        sessions_str, current_conversation_id
    )

    if not isinstance(conversations_list_output, tuple):
        conversations_list_output = (conversations_list_output,)

    return (
        conversations_list_output +
        render_current_conversation(
            sessions_str, current_conversation_id,
            **kwargs))  # type: ignore


def get_current_conversation_and_sessions(sessions_str, current_conversation_id):
    sessions: Any = json5.loads(sessions_str)

    if not current_conversation_id:
        return (None, sessions)

    if not isinstance(sessions, dict):
        return (None, None)

    if current_conversation_id not in sessions:
        return (None, sessions)

    return (sessions[current_conversation_id], sessions)


def render_current_conversation(sessions_str, current_conversation_id):
    current_conversation, _ = get_current_conversation_and_sessions(
        sessions_str, current_conversation_id)

    if not current_conversation:
        return (
            blank_conversation_header_content,
            gr.Box.update(visible=True),
            gr.Chatbot.update(value=[], visible=False),
            gr.Code.update(value=None),
            gr.Code.update(value=None),
        )

    prompter = get_prompter(current_conversation['prompt_template'])
    input_roles = prompter.get_input_roles()
    # output_roles = prompter.get_output_roles()

    chatbot_history = []

    for message in current_conversation.get('messages'):
        is_input = message['from'] in input_roles
        if not chatbot_history:
            if is_input:
                chatbot_history.append([message['message'], None])
            else:
                chatbot_history.append([None, message['message']])
            continue
        if is_input:
            chatbot_history.append([message['message'], None])
        else:
            if not chatbot_history[-1][1]:
                # Is a reply to the last message
                chatbot_history[-1][1] = message['message']
            else:
                # last message already have a reply, is a reply to nothing
                chatbot_history.append([None, message['message']])
    return (
        conversation_header_html(current_conversation),
        gr.Box.update(visible=False),
        gr.Chatbot.update(value=chatbot_history, visible=True),
        gr.Code.update(value=current_conversation.get('outputs') and current_conversation['outputs'].get('text')),
        gr.Code.update(value=current_conversation.get('outputs') and current_conversation['outputs'].get('tokens_str')),
    )


def pre_prepare_generate(
    sessions_str,
    current_conversation_id,
    model_preset,
    prompt_template,
):
    try:
        current_conversation, _ = get_current_conversation_and_sessions(
            sessions_str, current_conversation_id)
        if not current_conversation:
            # Only for temporary display usage, don't need to generate an ID.
            current_conversation = {
                'name': '',
                'model_preset': model_preset,
                'prompt_template': prompt_template,
                'conversations': [],
            }

        header_html = conversation_header_html(current_conversation)

        if not current_conversation.get('messages'):
            header_html += blank_conversation_header_content

        return (
            '',
            gr.Box.update(visible=False),
            gr.HTML.update(
                visible=True,
                value=header_html,
            ),
            gr.Textbox.update(),
        )

    except Exception as e:
        raise gr.Error(str(e) + '. Click the "Stop" button to dismiss this message.') from e


def prepare_generate(
    sessions_str,
    current_conversation_id,
    model_preset,
    prompt_template,
    message,
    # **kwargs,
):
    try:
        current_conversation, sessions = \
            get_current_conversation_and_sessions(
                sessions_str, current_conversation_id)
        if not sessions:
            sessions = {}
        sessions['_updated_at'] = get_time()
        if not current_conversation_id:
            current_conversation_id = get_new_conversation_id(sessions_str)
        if not current_conversation:
            existing_conversations_count = len([
                k for k in sessions.keys() if not k.startswith('_')])
            current_conversation = {
                'name': f"Conversation #{existing_conversations_count + 1}",
                'model_preset': model_preset,
                'prompt_template': prompt_template,
                'messages': [],
            }

        model_preset = get_model_preset_from_choice(model_preset)
        if model_preset:
            model_preset.tokenizer
            model_preset.model

        if message:
            prompter = get_prompter(prompt_template)
            current_conversation['messages'].append({
                'from': prompter.get_input_roles()[0],
                'message': message,
            })
        current_conversation['updated_at'] = get_time()

        sessions[current_conversation_id] = current_conversation

        if Config.ui_dev_mode:
            time.sleep(0.5)

        new_sessions_str = json.dumps(sessions, ensure_ascii=False)

        conversations_list_output = handle_update_conversations_list_html(
            new_sessions_str, current_conversation_id
        )

        # print('new_sessions_str returned by prepare', new_sessions_str)
        return (
            (
                '',
                new_sessions_str,
                current_conversation_id,
                gr.Textbox.update(value=''),
            )
            + conversations_list_output
            + render_current_conversation(**{
                # **kwargs,
                'sessions_str': new_sessions_str,
                'current_conversation_id': current_conversation_id,
            })
        )

    except Exception as e:
        raise gr.Error(str(e) + '. Click the "Stop" button to dismiss this message.') from e


def handle_generate(
    sessions_str,
    current_conversation_id,
    generation_config,
):
    # print('sessions_str got by handle_generate', sessions_str)
    try:
        # Need to yield at least once to prevent things being cleared out.
        yield (
            '',
            sessions_str,
        ) + render_current_conversation(**{
            # **kwargs,
            'sessions_str': sessions_str,
            'current_conversation_id': current_conversation_id,
        })


        current_conversation, sessions = \
            get_current_conversation_and_sessions(
                sessions_str, current_conversation_id)
        current_conversation: Any = current_conversation

        model_preset = get_model_preset_from_choice(
            current_conversation['model_preset'])

        prompt_template_name = current_conversation['prompt_template']
        prompter = get_prompter(prompt_template_name)
        if not prompter:
            raise ValueError(f"Can't find prompt template '{prompt_template_name}'")
        prompt = prompter.generate_dialogue_prompt_v1(
            current_conversation['messages'])
        stop_sequences = prompter.get_stop_sequences()

        generation_config = \
            prepare_generation_config_args(generation_config)
        generation_config = GenerationConfig(
            **generation_config,
        )

        output_message = None

        def get_output_message():
            nonlocal output_message
            if output_message:
                return output_message
            output_message = {
                'from': prompter.get_output_roles()[0],
                'message': '',
            }
            current_conversation['messages'].append(output_message)
            return output_message

        def set_output(name, value):
            nonlocal current_conversation
            if 'outputs' not in current_conversation:
                current_conversation['outputs'] = {}
            current_conversation['outputs'][name] = value

        def ui_generation_stopping_criteria(input_ids, score, **kwargs):
            if Global.should_stop_generating:
                return True
            return False

        Global.should_stop_generating = False

        model = model_preset.model
        tokenizer = model_preset.tokenizer
        generation_args = {
            'model': model,
            'tokenizer': tokenizer,
            'prompt': prompt,
            'generation_config': generation_config,
            'stopping_criteria': [ui_generation_stopping_criteria],
            'stop_sequences': stop_sequences,
            'stream_output': True
        }

        last_yield_at = None
        for (
            decoded_output, output, completed
        ) in Global.inference_generate_fn(**generation_args):
            if Global.should_stop_generating:
                return

            current_time = get_time()
            if not completed:
                # Throttling, rendering the chat UI is expensive
                # Yield every 0.1s
                if last_yield_at and current_time - last_yield_at < 100:
                    continue

            set_output('text', decoded_output)
            response = remove_common_from_start(prompt, decoded_output)

            get_output_message()['message'] = response

            output_tokens_str = comparing_lists(
                [
                    [tokenizer.decode([i]) for i in output],
                    [f"{i}," if c < len(output) - 1 else f"{i}"
                     for c, i in enumerate(output)],
                ],
                labels=['//', ''],
                max_width=80)
            output_tokens_str = output_tokens_str.rstrip()
            output_tokens_str = f"[\n{output_tokens_str}\n]"

            set_output('tokens_str', output_tokens_str)

            sessions['_updated_at'] = current_time
            new_sessions_str = json.dumps(sessions, ensure_ascii=False)
            last_yield_at = current_time

            yield (
                '',
                new_sessions_str,
            ) + render_current_conversation(**{
                # **kwargs,
                'sessions_str': new_sessions_str,
                'current_conversation_id': current_conversation_id,
            })
            # yield (
            #     gr.Textbox.update(value=response),
            #     output_tokens_str,
            #     gr.Textbox.update(
            #         value=get_output_for_flagging(
            #             decoded_output, output_tokens_str,
            #             completed=completed
            #         ),
            #         visible=True)
            # )

        # new_sessions_str = json.dumps(sessions, ensure_ascii=False)
        # print('new_sessions_str', new_sessions_str)
        # yield (
        #     '',
        #     new_sessions_str,
        # ) + render_current_conversation(**{
        #     # **kwargs,
        #     'sessions_str': new_sessions_str,
        #     'current_conversation_id': current_conversation_id,
        # })

    except Exception as e:
        raise gr.Error(str(e) + '. Click the "Stop" button to dismiss this message.') from e


def handle_stop_generate(sessions_str, current_conversation_id):
    Global.generation_force_stopped_at = time.time()
    Global.should_stop_generating = True

    current_conversation, _ = get_current_conversation_and_sessions(
        sessions_str, current_conversation_id)
    if not current_conversation:
        return (
            '',
            gr.Box.update(visible=True),
            blank_conversation_header_content,
            gr.HTML.update(),
            gr.Textbox.update(),
        )

    header_html = conversation_header_html(current_conversation)
    if not current_conversation.get('messages'):
        header_html += blank_conversation_header_content
    return (
        '',
        gr.Box.update(visible=False),
        header_html,
        gr.HTML.update(),
        gr.Textbox.update(),
    )


def get_new_conversation_id(sessions_str=None):
    random_bytes = os.urandom(16)
    hash_object = hashlib.sha256(random_bytes)
    hex_dig = hash_object.hexdigest()
    new_uid = hex_dig[:8]

    if sessions_str:
        sessions: Any = json5.loads(sessions_str)
        if sessions:
            if new_uid in sessions:
                # Generate a new uid if this one already exists.
                return get_new_conversation_id(sessions_str=sessions_str)

    return new_uid
