from typing import List

from textwrap import dedent


blank_conversation_header_content = '<div class="blank">Select a model and send a message to start the conversation</div>'


def conversation_list_item_html(data):
    info = []

    info.append(f"Model: {data.get('model_preset')}")
    info.append(f"Template: {data.get('prompt_template')}")

    # if (
    #     model_preset.tokenizer_name_or_path
    #     and model_preset.tokenizer_name_or_path != model_preset.model_name_or_path
    # ):
    #     info.append(
    #         f"Tokenizer: <code>{model_preset.tokenizer_name_or_path}</code>")

    # if model_preset.adapter_model_name_or_path:
    #     info.append(
    #         f"Adapter Model: <code>{model_preset.adapter_model_name_or_path}</code>")

    on_click = [
        f"document.querySelector('#chat_ui_current_conversation input').value = '{data.get('_id')}'",
        "document.querySelector('#chat_ui_current_conversation input').dispatchEvent(new Event('input', {'bubbles': true, 'cancelable': true }))",
        "document.getElementById('chat_ui_current_conversation_set').click()",
        "document.getElementById('chat_ui_current_conversation_column').scrollIntoView({ behavior: 'smooth' })"
    ]
    on_click = ';'.join(on_click)

    additional_classes = ''

    if data.get('_active'):
        additional_classes += ' is-active'

    html_content = dedent(
        f'''
        <div class="item{additional_classes}" onclick="{on_click}">
            <div class="name-container">
                <div class="name">
                    {data.get('name')}
                </div>
            </div>
            <div class="info">
                {'<br />'.join(info)}
            </div>
        </div>
        '''
    ).strip()
    return html_content


def conversation_list_html(conversations):
    sorted_conversations = sorted(
        conversations,
        key=lambda x: x.get('updated_at') or 0,
        reverse=True)

    items_html = [conversation_list_item_html(c) for c in sorted_conversations]
    items_html = '\n'.join(items_html)
    html_content = dedent(f'''
        <div class="chat-ui-conversations-list">
        {items_html}
        </div>
    ''').strip()
    return html_content


def conversation_header_html(conversation):
    html_content = f'''
        <div class="chat-ui-conversation-header">
            <div class="model-preset-container">
                <div class="title">Model</div>
                <div class="name">{conversation['model_preset']}</div>
            </div>
            <div class="prompt-template-container">
                <div class="title">Prompt Template</div>
                <div class="name">{conversation['prompt_template']}</div>
            </div>
        </div>
    '''
    return html_content

# def model_detail_html(
#     model_preset: ModelPreset,
#     default_preset_uid=None, starred_preset_uids=[]
# ):
#     html_content = ''

#     on_edit_click = [
#         f"document.querySelector('#models_model_preset_uid_to_edit input').value = '{model_preset.uid}'",
#         "document.querySelector('#models_model_preset_uid_to_edit input').dispatchEvent(new Event('input', {'bubbles': true, 'cancelable': true }))",
#         "document.getElementById('models_edit_model_preset_btn').click()",
#     ]
#     on_edit_click = ';'.join(on_edit_click)

#     on_duplicate_click = [
#         f"document.querySelector('#models_model_preset_uid_to_duplicate input').value = '{model_preset.uid}'",
#         "document.querySelector('#models_model_preset_uid_to_duplicate input').dispatchEvent(new Event('input', {'bubbles': true, 'cancelable': true }))",
#         "document.getElementById('models_duplicate_model_preset_btn').click()",
#     ]
#     on_duplicate_click = ';'.join(on_duplicate_click)

#     on_delete_click = [
#         f"document.querySelector('#models_model_preset_uid_to_delete input').value = '{model_preset.uid}'",
#         "document.querySelector('#models_model_preset_uid_to_delete input').dispatchEvent(new Event('input', {'bubbles': true, 'cancelable': true }))",
#         "document.getElementById('models_delete_model_preset_btn').click()",
#     ]
#     on_delete_click = ';'.join(on_delete_click)
#     on_delete_click = dedent(f"""
#         if (confirm('Are you sure you want to delete the preset \\'{model_preset.name}\\' ({model_preset.uid})? This cannot be reverted.')) {{
#             {on_delete_click}
#         }}
#     """).strip().replace('\n', ' ')

#     on_set_as_default_click = [
#         f"document.querySelector('#models_model_preset_uid_to_set_as_default input').value = '{model_preset.uid}'",
#         "document.querySelector('#models_model_preset_uid_to_set_as_default input').dispatchEvent(new Event('input', {'bubbles': true, 'cancelable': true }))",
#         "document.getElementById('models_set_as_default_model_preset_btn').click()",
#     ]
#     on_set_as_default_click = ';'.join(on_set_as_default_click)

#     on_star_click = [
#         f"document.querySelector('#models_model_preset_uid_to_toggle_star input').value = '{model_preset.uid}'",
#         "document.querySelector('#models_model_preset_uid_to_toggle_star input').dispatchEvent(new Event('input', {'bubbles': true, 'cancelable': true }))",
#         "document.getElementById('models_toggle_model_preset_star_btn').click()",
#     ]
#     on_star_click = ';'.join(on_star_click)

#     html_content += '<div class="models-ui-block-title-and-actions">'
#     html_content += f"<h2>{model_preset.name}</h2>"

#     html_content += dedent(f'''
#         <div class="models-ui-block-actions">
#             <button onclick="{on_delete_click}">Delete</button>
#     ''').strip()

#     html_content += dedent(f'''
#         <button onclick="{on_duplicate_click}">Duplicate</button>
#     ''').strip()

#     if model_preset.uid != default_preset_uid:
#         html_content += dedent(f'''
#             <button onclick="{on_set_as_default_click}">Set as Default</button>
#         ''').strip()

#     if model_preset.uid in starred_preset_uids:
#         html_content += dedent(f'''
#             <button onclick="{on_star_click}">★</button>
#         ''').strip()
#     else:
#         html_content += dedent(f'''
#             <button onclick="{on_star_click}">☆</button>
#         ''').strip()

#     html_content += dedent(f'''
#             <button onclick="{on_edit_click}">Edit</button>
#         </div>
#     ''').strip()

#     html_content += '</div>'

#     html_content += '<div class="models-ui-block-details">'

#     html_content += '<h3>Model</h3>'

#     model_tags = ''
#     if model_preset.load_model_from == 'data_dir':
#         model_tags += '&nbsp;&nbsp;<span class="tag">Local</span>'
#     html_content += f'<p class="ph-lg">{model_preset.model_name_or_path}{model_tags}</p>'

#     # html_content += f'''
#     # <ul>
#     #     <li>Torch dtype: <code>...</code></li>
#     # </ul>
#     # '''

#     html_content += '</div>'

#     return html_content
