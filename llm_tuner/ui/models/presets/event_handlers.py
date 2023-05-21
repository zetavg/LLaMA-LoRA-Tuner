from typing import Any, Dict

import time
import json
import traceback
import gradio as gr
from textwrap import dedent

from ....data import (
    get_model_presets,
    get_model_preset,
    get_new_model_preset,
    save_model_preset,
    delete_model_preset,
    get_model_preset_settings,
    update_model_preset_settings,
)

from .html_templates import model_preset_list_item_html


def handle_load_model_presets():
    try:
        model_presets = get_model_presets()
        items_html = [model_preset_list_item_html(p) for p in model_presets]
        items_html = '\n'.join(items_html)
        html_content = dedent(f'''
            <div class="list">
            {items_html}
            </div>
        ''').strip()

        return (
            gr.HTML.update(value=html_content),
            gr.HTML.update(visible=False),
        )
    except Exception as e:
        raise gr.Error(str(e)) from e


def handle_show_model_preset(preset_uid):
    try:
        model_preset_settings = get_model_preset_settings()
        default_preset_uid = model_preset_settings.get('default_preset_uid')

        model_preset = get_model_preset(preset_uid)
        html_content = ''

        on_edit_click = [
            f"document.querySelector('#models_model_preset_uid_to_edit input').value = '{model_preset.uid}'",
            "document.querySelector('#models_model_preset_uid_to_edit input').dispatchEvent(new Event('input', {'bubbles': true, 'cancelable': true }))",
            "document.getElementById('models_edit_model_preset_btn').click()",
        ]
        on_edit_click = ';'.join(on_edit_click)

        on_delete_click = [
            f"document.querySelector('#models_model_preset_uid_to_delete input').value = '{model_preset.uid}'",
            "document.querySelector('#models_model_preset_uid_to_delete input').dispatchEvent(new Event('input', {'bubbles': true, 'cancelable': true }))",
            "document.getElementById('models_delete_model_preset_btn').click()",
        ]
        on_delete_click = ';'.join(on_delete_click)
        on_delete_click = dedent(f"""
            if (confirm('Are you sure you want to delete the preset \\'{model_preset.name}\\' ({model_preset.uid})? This cannot be reverted.')) {{
                {on_delete_click}
            }}
        """).strip().replace('\n', ' ')

        on_set_as_default_click = [
            f"document.querySelector('#models_model_preset_uid_to_set_as_default input').value = '{model_preset.uid}'",
            "document.querySelector('#models_model_preset_uid_to_set_as_default input').dispatchEvent(new Event('input', {'bubbles': true, 'cancelable': true }))",
            "document.getElementById('models_set_as_default_model_preset_btn').click()",
        ]
        on_set_as_default_click = ';'.join(on_set_as_default_click)

        html_content += dedent(f'''
            <div class="models-ui-block-actions">
                <button onclick="{on_delete_click}">Delete</button>
        ''').strip()

        if model_preset.uid != default_preset_uid:
            html_content += dedent(f'''
                <button onclick="{on_set_as_default_click}">Set as Default</button>
            ''').strip()

        html_content += dedent(f'''
                <button onclick="{on_edit_click}">Edit</button>
            </div>
        ''').strip()

        html_content += f"<h2>{model_preset.name}</h2>"

        return (
            gr.Column.update(visible=False),
            gr.Column.update(visible=True),
            gr.HTML.update(html_content),
            gr.Code.update(value=json.dumps(
                model_preset.data, indent=2, ensure_ascii=False)),
        )
    except Exception as e:
        raise gr.Error(str(e)) from e


def handle_edit_model_preset(preset_uid):
    try:
        p = get_model_preset(preset_uid)
        return handle_edit_model_preset_by_data(p.data)
    except Exception as e:
        raise gr.Error(str(e)) from e


def handle_delete_model_preset(preset_uid):
    try:
        delete_model_preset(preset_uid)
        return handle_load_model_presets() + (
            gr.Column.update(visible=True),
            gr.Column.update(visible=True),
            gr.Box.update(visible=False),
            gr.Column.update(visible=False),
        )
    except Exception as e:
        raise gr.Error(str(e)) from e


def handle_new_model_preset():
    try:
        data = get_new_model_preset()
        return handle_edit_model_preset_by_data(data)
    except Exception as e:
        raise gr.Error(str(e)) from e


def handle_edit_model_preset_by_data(data: Dict[str, Any]):
    try:
        uid = data['_uid']
        file_name = data['_file_name']
        del data['_uid']
        del data['_file_name']
        title_content = \
            f"<h2>Editing {data['name']} (<code>{uid}</code>)</h2>"
        if not file_name:
            title_content = f"<h2>Create new preset (<code>{uid}</code>)</h2>"
        return (
            gr.Row.update(visible=False),
            gr.Box.update(visible=True),
            gr.HTML.update(title_content),
            uid,
            file_name,
            gr.Code.update(value=json.dumps(
                data, indent=2, ensure_ascii=False)),
        )
    except Exception as e:
        raise gr.Error(str(e)) from e


def handle_save_edit(uid, original_file_name, json_str):
    try:
        data = json.loads(json_str)
        save_model_preset(uid, data, original_file_name=original_file_name)
        return (
            gr.HTML.update(visible=False),
            uid,
            gr.Row.update(visible=True),
            gr.Box.update(visible=False),
        )
    except Exception as e:
        traceback.print_exc()
        print(e)
        return (
            gr.HTML.update(
                value=f'<div class="error">{str(e)}</div>', visible=True),
            uid,
            gr.Row.update(visible=False),
            gr.Box.update(visible=True),
        )


def handle_discard_edit():
    return (
        gr.Row.update(visible=True),
        gr.Box.update(visible=False),
    )


def handle_set_model_preset_as_default(uid):
    update_model_preset_settings({'default_preset_uid': uid})
    return handle_load_model_presets() + handle_show_model_preset(uid)
