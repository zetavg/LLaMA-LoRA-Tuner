from typing import List, Tuple, Union, Dict

import os
import hashlib
from textwrap import dedent

import gradio as gr


ControlDefn = Tuple[
    Union[gr.Textbox, gr.Slider, gr.Checkbox, gr.Dropdown],
    Union[str, List[str]],
    Union[str, Dict[str, str]],
]

ControlsDefn = List[ControlDefn]


def tie_controls_with_json_editor(
    json_editor: gr.Code,
    controls_defn: ControlsDefn,
    message_component: gr.HTML,
    status_indicator_elem_id: Union[str, None] = None,
):
    uid = get_random_hex()
    wx_btn_elem_id = f"wx_tie_controls_with_json_editor_{uid}_btn"

    # Update the code block on controls change
    for (component, path, t) in controls_defn:
        if not isinstance(path, list):
            path = [path]

        js_code_to_assign_value = ''
        if t == 'number':
            js_code_to_assign_value = dedent(f"""
                {get_js_code_to_ensure_path_can_be_accessed('json', path)}
                value = parseFloat(value, 10);
                {get_js_property_accessor('json', path)} = value;
            """).strip()
        elif t == 'string':
            js_code_to_assign_value = dedent(f"""
                if (value || {get_safe_js_property_accessor('json', path)}) {{
                    {get_js_code_to_ensure_path_can_be_accessed('json', path)}
                    value = (value || '') + '';
                    {get_js_property_accessor('json', path)} = value;
                }}
            """).strip()
        elif t == 'boolean':
            js_code_to_assign_value = dedent(f"""
                {get_js_code_to_ensure_path_can_be_accessed('json', path)}
                if (value === 'Yes') {{
                    value = true;
                }} else if (value === 'No') {{
                    value = false;
                }} else if (value === 'Default') {{
                    value = null;
                }}
                {get_js_property_accessor('json', path)} = value;
            """).strip()
        elif isinstance(t, dict):
            js_code_to_assign_value = dedent(f"""
                {get_js_code_to_ensure_path_can_be_accessed('json', path)}
                var v = null;
                switch (value) {{
            """).strip()
            for k, v in t.items():
                js_code_to_assign_value += dedent(f"""
                    case '{k}':
                        v = {v};
                        break;
                """).strip()
            js_code_to_assign_value += dedent(f"""
                }}

                {get_js_property_accessor('json', path)} = v;
            """).strip()
        else:
            raise ValueError(f"Unknown type: '{t}")

        component.change(
            fn=None,
            _js=dedent(f"""
            function (value, json_string) {{
                try {{
                    if (window.debug_tie_controls_with_json_editor) {{
                        debugger
                    }}
                    var json = JSON.parse(json_string);
                    {js_code_to_assign_value}
                    return [value, JSON.stringify(json, null, 2)];
                }} catch (e) {{
                    console.log(e);
                    alert("Cannot update value: " + e);
                    return [value, json_string];
                }} finally {{
                    // This is a workaround to resolve that if the page has been
                    // loaded for too long, the UI will stop reflecting the updates
                    // (while the value update is actually done, and any subsequent
                    // server-side code will be executed with the updated value and
                    // make it update on the UI).
                    if (window['timer_{wx_btn_elem_id}']) {{
                        clearTimeout(window['timer_{wx_btn_elem_id}']);
                    }}
                    window['timer_{wx_btn_elem_id}'] = setTimeout(function () {{
                        var btn = document.getElementById('{wx_btn_elem_id}')
                        if (btn) {{
                            btn.click();
                        }} else {{
                            console.error('Cannot find button with id "{wx_btn_elem_id}"');
                        }}
                    }}, 321)
                }}
            }}
            """).strip(),
            inputs=[component, json_editor],
            outputs=[component, json_editor],
        )

    # Update controls on code block change
    # js_code = []
    # for (component, path, t) in controls_defn:
    #     components.append(component)

    #     if not isinstance(path, list):
    #         path = [path]

    #     js_code = f"""
    #         values.push(json.{val_path})
    #     """

    #     js_code_to_add_updated_value.append(
    #         js_code_to_ensure_path_accessability + js_code
    #     )
    # js_code_to_add_updated_value = ';'.join(js_code_to_add_updated_value)
    js_code = dedent(f"""
        function () {{
            try {{
                json_string = arguments[0];
                if (!json_string) return [];
                var json = JSON.parse(json_string);
                var jsonShapeOfValuesThatCanBeControlled = {{}};
                oldValues = Array.from(arguments).slice(1);
                values = [];
    """).strip()
    for (component, path, t) in controls_defn:
        if not isinstance(path, list):
            path = [path]

        js_code += get_js_code_to_ensure_path_can_be_accessed(
            'jsonShapeOfValuesThatCanBeControlled', path
        ) + '\n'
        js_shape_property_accessor = \
            get_js_property_accessor(
                'jsonShapeOfValuesThatCanBeControlled', path
            )
        js_code += f"{js_shape_property_accessor} = true;\n"

        js_code += get_js_code_to_ensure_path_can_be_accessed(
            'json', path) + '\n'
        if t == 'boolean' and component.__class__ == gr.Dropdown:
            js_code += f'''
                var v = {get_js_property_accessor('json', path)};
                if (v === true) {{
                    values.push('Yes');
                }} else if (v === false) {{
                    values.push('No');
                }} else {{
                    values.push('Default');
                }}
            '''
        elif t == 'string' and component.__class__ == gr.Dropdown:
            null_value = 'undefined'
            if 'None' in component.choices:  # type: ignore
                null_value = "'None'"
            js_code += f'''
                var v = {get_js_property_accessor('json', path)};
                if (v === null || v === undefined) {{
                    values.push({null_value});
                }} else {{
                    values.push(v);
                }}
            '''
        elif isinstance(t, dict):
            js_code += f'''
                var v = {get_js_property_accessor('json', path)};
                switch (v) {{
            '''
            default_value = 'undefined'
            for k, v in t.items():
                if v == "'default'" or v == '"default"':
                    default_value = k
                js_code += f'''
                        case {v}:
                            values.push('{k}');
                            break;
                '''
            js_code += f'''
                    default:
                        values.push('{default_value}');
                        break;
                }}
            '''
        else:
            js_code += f"values.push({get_js_property_accessor('json', path)});\n"
    js_code += dedent(f"""
                if (window.debug_tie_controls_with_json_editor) {{
                    debugger
                }}
                var getDifference = {get_js_get_difference_function_code()};
                var difference = getDifference(json, jsonShapeOfValuesThatCanBeControlled);
                if (difference.length > 0) {{
                    console.log('difference', difference);
                }}
    """).strip()
    if status_indicator_elem_id:
        js_code += dedent(f"""
            if (difference.length > 0) {{
                document.getElementById('{status_indicator_elem_id}').classList.add('has-difference');
            }} else {{
                document.getElementById('{status_indicator_elem_id}').classList.remove('has-difference');
            }}
        """).strip()
    if status_indicator_elem_id:
        js_code += dedent(f"""
            document.getElementById('{status_indicator_elem_id}').classList.remove('has-error');
        """).strip()
    js_code += dedent(f"""
                return [''].concat(values);
            }} catch (e) {{
                console.error(e);
    """).strip()
    if status_indicator_elem_id:
        js_code += dedent(f"""
            document.getElementById('{status_indicator_elem_id}').classList.add('has-error');
        """).strip()
    js_code += dedent(f"""
                return [
                    '<div class="json-code-block-error-message">' + e + '</div>'
                ];
            }} finally {{
                // This is a workaround to resolve that if the page has been
                // loaded for too long, the UI will stop reflecting the updates
                // (while the value update is actually done, and any subsequent
                // server-side code will be executed with the updated value and
                // make it update on the UI).
                if (window['timer_{wx_btn_elem_id}']) {{
                    clearTimeout(window['timer_{wx_btn_elem_id}']);
                }}
                window['timer_{wx_btn_elem_id}'] = setTimeout(function () {{
                    var btn = document.getElementById('{wx_btn_elem_id}')
                    if (btn) {{
                        btn.click();
                    }} else {{
                        console.error('Cannot find button with id "{wx_btn_elem_id}"');
                    }}
                }}, 321)
            }}
        }}
    """).strip()
    control_components = [d[0] for d in controls_defn]
    json_editor.change(
        fn=None,
        _js=js_code,
        inputs=[json_editor] + control_components,
        outputs=[message_component] + control_components,
    )

    with gr.Blocks() as blocks:
        with gr.Column(
            visible=False,
            elem_id=f"wx_tie_controls_with_json_editor_{uid}_container",
        ):
            wx_textbox = gr.Textbox(
                label="Textbox for 'tie_controls_with_json_editor' to work",
                elem_id=f"wx_tie_controls_with_json_editor_{uid}_textbox"
            )
            wx_btn = gr.Button(
                "Button for 'tie_controls_with_json_editor' to work",
                elem_id=wx_btn_elem_id
            )
            wx_btn.click(
                fn=wx_value_updater,
                inputs=[wx_textbox],
                outputs=[wx_textbox]
            )

    blocks.load(
        fn=None,
        _js=js_code,
        inputs=[json_editor] + control_components,
        outputs=[message_component] + control_components,
    )


def get_js_code_to_ensure_path_can_be_accessed(
    object_name: str,
    path: List[str]
) -> str:
    codes = []
    for i in range(len(path) - 1):
        js_p = '.'.join(path[:i + 1])
        codes.append(
            f"if (!{object_name}.{js_p}) {{ {object_name}.{js_p} = {{}} }}"
        )
    return ' '.join(codes)


def get_js_property_accessor(
    object_name: str,
    path: List[str]
) -> str:
    return f"{object_name}.{'.'.join(path)}"


def get_safe_js_property_accessor(
    object_name: str,
    path: List[str]
) -> str:
    code = [object_name]
    for i in range(len(path)):
        code.append('.'.join([object_name] + path[:i + 1]))
    return ' && '.join(code)


def get_js_get_difference_function_code() -> str:
    return dedent("""
        function (a, b, prefix = '') {
          let diff = [];

          for (let key in a) {
            if (a.hasOwnProperty(key)) {
              if (b.hasOwnProperty(key)) {
                if (typeof a[key] === 'object' && typeof b[key] === 'object') {
                  diff = diff.concat(getDifference(a[key], b[key], prefix + key + '.'));
                }
              } else {
                diff.push(prefix + key);
              }
            }
          }

          return diff;
        }
    """).strip()


def get_random_hex():
    random_bytes = os.urandom(16)
    hash_object = hashlib.sha256(random_bytes)
    hex_dig = hash_object.hexdigest()
    return hex_dig


def wx_value_updater(x):
    try:
        x = int(x)
        x += 1
        if x >= 100:
            x = 0
        return x
    except Exception:
        return '0'
