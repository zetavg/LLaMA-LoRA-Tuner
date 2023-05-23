import json
from textwrap import dedent
import gradio as gr

from ...config import Config
from ..css_styles import register_css_style
from ..ui_utils import tie_controls_with_json_editor
from ...utils.data_processing import deep_merge_dicts


def generation_options(
    value='{}',
    elem_id_prefix=None,
    elem_classes=None,
    # status_indicator_elem_id=None
):
    if not isinstance(value, dict):
        value = json.loads(value)

    value = deep_merge_dicts(Config.default_generation_config, value)

    def get_elem_id(elem_id):
        if not elem_id_prefix:
            return None
        return '_'.join([elem_id_prefix, elem_id])

    with gr.Column(
        elem_classes=f"generation-options-container accordion-container {elem_classes}",
        elem_id=get_elem_id('generation_options_container'),
    ):
        with gr.Column(elem_classes="generation-config-controls"):
            temperature = gr.Slider(
                label="Temperature",
                minimum=0, maximum=2, step=0.01,
                elem_classes='temperature',
                elem_id=get_elem_id('temperature')
            )

            with gr.Row():
                top_k = gr.Slider(
                    label="Top K",
                    minimum=0, maximum=100, step=1,
                    elem_classes='top_k',
                    elem_id=get_elem_id('top_k')
                )
                top_p = gr.Slider(
                    label="Top P",
                    minimum=0, maximum=1, step=0.01,
                    elem_classes='top_p',
                    elem_id=get_elem_id('top_p')
                )

            num_beams = gr.Slider(
                label="Beams",
                minimum=1, maximum=5, step=1,
                elem_classes='num_beams',
                elem_id=get_elem_id('num_beams')
            )

            repetition_penalty = gr.Slider(
                label="Repetition Penalty",
                minimum=0, maximum=2.5, step=0.01,
                elem_classes='repetition_penalty',
                elem_id=get_elem_id('repetition_penalty')
            )

            max_new_tokens = gr.Slider(
                label="Max New Tokens",
                minimum=1, maximum=2048, step=1,
                elem_classes='max_new_tokens',
                elem_id=get_elem_id('max_new_tokens')
            )

            do_sample = gr.Checkbox(
                label="Do Sample",
                value=False,
                visible=False,
                elem_classes='do_sample',
                elem_id=get_elem_id('do_sample')
            )

        with gr.Accordion(
            "Advanced Generation Config", open=False,
            elem_id=get_elem_id('advanced_generation_config_accordion'),
            elem_classes='accordion accordion-with-block-title-text-color',
        ):
            generation_config_json = gr.Code(
                label="Generation Config (JSON)",
                language='json',
                value=json.dumps(value, indent=2, ensure_ascii=False),
                elem_id=get_elem_id('generation_config_json'),
                elem_classes='generation_config_json',
            )
            generation_config_json_message = gr.HTML('')
            gr.Markdown(
                elem_classes='info-text mt-sm ph-md',
                value=dedent(f"""
                   See the [docs](https://huggingface.co/docs/transformers/v4.28.1/en/main_classes/text_generation#transformers.GenerationConfig) for more information.
                """).strip()
            )

        tie_controls_with_json_editor(
            generation_config_json,
            [
                (temperature, 'temperature', 'number'),
                (top_p, 'top_p', 'number'),
                (top_k, 'top_k', 'number'),
                (num_beams, 'num_beams', 'number'),
                (repetition_penalty, 'repetition_penalty', 'number'),
                (max_new_tokens, 'max_new_tokens', 'number'),
                (do_sample, 'do_sample', 'boolean'),
            ],
            generation_config_json_message,
            status_indicator_elem_id=get_elem_id(
                'generation_options_container'),
        )

        temperature.change(
            fn=None,
            _js=dedent("""
                function (temperature, generation_config_json) {
                    var generation_config = JSON.parse(generation_config_json);
                    generation_config.temperature = temperature;
                    if (temperature > 0) {
                        generation_config.do_sample = true;
                    } else {
                        generation_config.do_sample = false;
                    }
                    return [JSON.stringify(generation_config, null, 2)];
                }
            """).strip(),
            inputs=[temperature, generation_config_json],
            outputs=[generation_config_json],
        )

    return {
        'generation_config_json': generation_config_json,
        'temperature': temperature,
        'top_p': top_p,
        'top_k': top_k,
        'num_beams': num_beams,
        'repetition_penalty': repetition_penalty,
        'max_new_tokens': max_new_tokens,
    }


register_css_style(
    'generation_options_component',
    '''
    .generation-options-container .form,
    .generation-options-container .block:not(.generation_config_json) {
        border: 0 !important;
        border-radius: 0 !important;
    }

    .generation-options-container {
        background-color: var(--border-color-primary);
    }
    .generation-options-container,
    .generation-options-container .form,
    .generation-options-container .generation-config-controls {
        gap: var(--block-border-width) !important;
    }

    .generation-options-container .accordion .gap {
        gap: 0 !important;
    }

    .generation-options-container .accordion::before {
        border-radius: 0 !important;
    }

    .generation-options-container.has-error .generation-config-controls::after {
        content: 'Please fix the errors in Advanced Generation Config below';
        display: block;
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        z-index: 40;
        background-color: var(--border-color-primary);
        opacity: 0.8;
        padding: var(--spacing-xxl);
        display: flex;
        justify-content: center;
        align-items: center;
        text-align: center;
        text-transform: uppercase;
    }

    .generation-options-container .temperature,
    .generation-options-container .top_p,
    .generation-options-container .top_k,
    .generation-options-container .num_beams,
    .generation-options-container .repetition_penalty,
    .generation-options-container .max_new_tokens,
    .generation-options-container .do_sample {
        padding-bottom: var(--spacing-md);
    }
    '''
)
