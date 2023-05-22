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

from .html_templates import (
    model_preset_list_item_html,
    model_preset_list_html,
    model_detail_html,
)


def handle_load_model_presets():
    try:
        model_presets = get_model_presets()
        html_content = model_preset_list_html(model_presets)

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
        starred_preset_uids = model_preset_settings.get(
            'starred_preset_uids', [])

        model_preset = get_model_preset(preset_uid)

        html_content = model_detail_html(
            model_preset,
            default_preset_uid=default_preset_uid,
            starred_preset_uids=starred_preset_uids,
        )

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


def handle_duplicate_model_preset(preset_uid):
    try:
        p = get_model_preset(preset_uid)
        data = p.data.copy()
        data['_uid'] = get_new_model_preset()['_uid']
        data['_file_name'] = None
        data['name'] += ' - Copy'
        return handle_edit_model_preset_by_data(data)
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
        data = {k: v for k, v in data.items() if not k.startswith('_')}
        title_content = \
            f"<h2>Editing {data['name']} (<code>{uid}</code>)</h2>"
        if not file_name:
            title_content = f"<h2>Create new preset (<code>{uid}</code>)</h2>"
        return (
            gr.Row.update(visible=False),
            gr.Box.update(visible=True),
            gr.HTML.update(title_content),
            gr.HTML.update('', visible=False),
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


def handle_toggle_model_preset_star(uid):
    settings = get_model_preset_settings()
    starred_preset_uids = settings.get('starred_preset_uids', [])
    if uid in starred_preset_uids:
        starred_preset_uids = [i for i in starred_preset_uids if i != uid]
    else:
        starred_preset_uids.append(uid)

    update_model_preset_settings({'starred_preset_uids': starred_preset_uids})
    return handle_load_model_presets() + handle_show_model_preset(uid)
