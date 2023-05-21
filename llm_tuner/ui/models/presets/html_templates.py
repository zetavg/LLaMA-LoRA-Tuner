from typing import List

from textwrap import dedent


from ....dataclasses import ModelPreset


def model_preset_list_item_html(model_preset: ModelPreset):
    info = []

    info.append(f"Model: <code>{model_preset.model_name_or_path}</code>")

    if (
        model_preset.tokenizer_name_or_path
        and model_preset.tokenizer_name_or_path != model_preset.model_name_or_path
    ):
        info.append(
            f"Tokenizer: <code>{model_preset.tokenizer_name_or_path}</code>")

    if model_preset.adapter_model_name_or_path:
        info.append(
            f"Adapter Model: <code>{model_preset.adapter_model_name_or_path}</code>")

    on_click = [
        f"document.querySelector('#models_model_preset_uid_to_show input').value = '{model_preset.uid}'",
        "document.querySelector('#models_model_preset_uid_to_show input').dispatchEvent(new Event('input', {'bubbles': true, 'cancelable': true }))",
        "document.getElementById('models_show_model_preset_btn').click()",
        "document.getElementById('models_right_column_container').scrollIntoView({ behavior: 'smooth' })"
    ]
    on_click = ';'.join(on_click)

    additional_classes = ''

    if model_preset.data.get('_is_default'):
        additional_classes += ' is-default'

    if model_preset.data.get('_is_starred'):
        additional_classes += ' is-starred'

    html_content = dedent(
        f'''
        <div class="item{additional_classes}" onclick="{on_click}">
            <div class="name-and-uid">
                <div class="name">
                    {model_preset.name}
                </div>
                <code class="uid">
                    {model_preset.uid}
                </code>
            </div>
            <div class="info">
                {'<br />'.join(info)}
            </div>
        </div>
        '''
    ).strip()
    return html_content


def model_preset_list_html(model_presets: List[ModelPreset]):
    items_html = [model_preset_list_item_html(p) for p in model_presets]
    items_html = '\n'.join(items_html)
    html_content = dedent(f'''
        <div class="list">
        {items_html}
        </div>
    ''').strip()
    return html_content
