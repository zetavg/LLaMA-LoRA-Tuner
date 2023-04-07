import gradio as gr
import time

import torch
import transformers
from transformers import GenerationConfig

from ..globals import Global
from ..models import get_base_model, get_model_with_lora, get_tokenizer, get_device
from ..utils.data import (
    get_available_template_names,
    get_available_lora_model_names,
    get_path_of_available_lora_model)
from ..utils.prompter import Prompter
from ..utils.callbacks import Iteratorize, Stream

device = get_device()

default_show_raw = True


def do_inference(
    lora_model_name,
    prompt_template,
    variable_0, variable_1, variable_2, variable_3,
    variable_4, variable_5, variable_6, variable_7,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    repetition_penalty=1.2,
    max_new_tokens=128,
    stream_output=False,
    show_raw=False,
    progress=gr.Progress(track_tqdm=True),
):
    try:
        variables = [variable_0, variable_1, variable_2, variable_3,
                     variable_4, variable_5, variable_6, variable_7]
        prompter = Prompter(prompt_template)
        prompt = prompter.generate_prompt(variables)

        if lora_model_name is not None and "/" not in lora_model_name and lora_model_name != "None":
            path_of_available_lora_model = get_path_of_available_lora_model(
                lora_model_name)
            if path_of_available_lora_model:
                lora_model_name = path_of_available_lora_model

        if Global.ui_dev_mode:
            message = f"Currently in UI dev mode, not running actual inference.\n\nLoRA model: {lora_model_name}\n\nYour prompt is:\n\n{prompt}"
            print(message)
            time.sleep(1)
            yield message, '[0]'
            return

        model = get_base_model()
        if not lora_model_name == "None" and lora_model_name is not None:
            model = get_model_with_lora(lora_model_name)
        tokenizer = get_tokenizer()

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            num_beams=num_beams,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

        if stream_output:
            # Stream the reply 1 token at a time.
            # This is based on the trick of using 'stopping_criteria' to create an iterator,
            # from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.

            def generate_with_callback(callback=None, **kwargs):
                kwargs.setdefault(
                    "stopping_criteria", transformers.StoppingCriteriaList()
                )
                kwargs["stopping_criteria"].append(
                    Stream(callback_func=callback)
                )
                with torch.no_grad():
                    model.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                return Iteratorize(
                    generate_with_callback, kwargs, callback=None
                )

            with generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    # new_tokens = len(output) - len(input_ids[0])
                    decoded_output = tokenizer.decode(output)

                    if output[-1] in [tokenizer.eos_token_id]:
                        break

                    raw_output = None
                    if show_raw:
                        raw_output = str(output)
                    yield prompter.get_response(decoded_output), raw_output
            return  # early return for stream_output

        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        raw_output = None
        if show_raw:
            raw_output = str(s)
        yield prompter.get_response(output), raw_output

    except Exception as e:
        raise gr.Error(e)


def reload_selections(current_lora_model, current_prompt_template):
    available_template_names = get_available_template_names()
    available_template_names_with_none = available_template_names + ["None"]

    if current_prompt_template not in available_template_names_with_none:
        current_prompt_template = None

    current_prompt_template = current_prompt_template or next(
        iter(available_template_names_with_none), None)

    default_lora_models = ["tloen/alpaca-lora-7b"]
    available_lora_models = default_lora_models + get_available_lora_model_names()
    available_lora_models = available_lora_models + ["None"]

    current_lora_model = current_lora_model or next(
        iter(available_lora_models), None)

    return (gr.Dropdown.update(choices=available_lora_models, value=current_lora_model),
            gr.Dropdown.update(choices=available_template_names_with_none, value=current_prompt_template))


def handle_prompt_template_change(prompt_template):
    prompter = Prompter(prompt_template)
    var_names = prompter.get_variable_names()
    human_var_names = [' '.join(word.capitalize()
                                for word in item.split('_')) for item in var_names]
    gr_updates = [gr.Textbox.update(
        label=name, visible=True) for name in human_var_names]
    while len(gr_updates) < 8:
        gr_updates.append(gr.Textbox.update(
            label="Not Used", visible=False))
    return gr_updates


def update_prompt_preview(prompt_template,
                          variable_0, variable_1, variable_2, variable_3,
                          variable_4, variable_5, variable_6, variable_7):
    variables = [variable_0, variable_1, variable_2, variable_3,
                 variable_4, variable_5, variable_6, variable_7]
    prompter = Prompter(prompt_template)
    prompt = prompter.generate_prompt(variables)
    return gr.Textbox.update(value=prompt)


def inference_ui():
    things_that_might_timeout = []

    with gr.Blocks() as inference_ui_blocks:
        with gr.Row():
            lora_model = gr.Dropdown(
                label="LoRA Model",
                elem_id="inference_lora_model",
                value="tloen/alpaca-lora-7b",
                allow_custom_value=True,
            )
            prompt_template = gr.Dropdown(
                label="Prompt Template",
                elem_id="inference_prompt_template",
            )
            reload_selections_button = gr.Button(
                "↻",
                elem_id="inference_reload_selections_button"
            )
            reload_selections_button.style(
                full_width=False,
                size="sm")
        with gr.Row():
            with gr.Column():
                with gr.Column(elem_id="inference_prompt_box"):
                    variable_0 = gr.Textbox(
                        lines=2,
                        label="Prompt",
                        placeholder="Tell me about alpecas and llamas.",
                        elem_id="inference_variable_0"
                    )
                    variable_1 = gr.Textbox(
                        lines=2, label="", visible=False, elem_id="inference_variable_1")
                    variable_2 = gr.Textbox(
                        lines=2, label="", visible=False, elem_id="inference_variable_2")
                    variable_3 = gr.Textbox(
                        lines=2, label="", visible=False, elem_id="inference_variable_3")
                    variable_4 = gr.Textbox(
                        lines=2, label="", visible=False, elem_id="inference_variable_4")
                    variable_5 = gr.Textbox(
                        lines=2, label="", visible=False, elem_id="inference_variable_5")
                    variable_6 = gr.Textbox(
                        lines=2, label="", visible=False, elem_id="inference_variable_6")
                    variable_7 = gr.Textbox(
                        lines=2, label="", visible=False, elem_id="inference_variable_7")

                    with gr.Accordion("Preview", open=False, elem_id="inference_preview_prompt_container"):
                        preview_prompt = gr.Textbox(
                            show_label=False, interactive=False, elem_id="inference_preview_prompt")
                        update_prompt_preview_btn = gr.Button(
                            "↻", elem_id="inference_update_prompt_preview_btn", full_width=False)
                        update_prompt_preview_btn.style(size="sm")

                # with gr.Column():
                #     with gr.Row():
                #         generate_btn = gr.Button(
                #             "Generate", variant="primary", label="Generate", elem_id="inference_generate_btn",
                #         )
                #         stop_btn = gr.Button(
                #             "Stop", variant="stop", label="Stop Iterating", elem_id="inference_stop_btn")

                # with gr.Column():
                with gr.Accordion("Options", open=True, elem_id="inference_options_accordion"):
                    temperature = gr.Slider(
                        minimum=0, maximum=1, value=0.1, step=0.01,
                        label="Temperature",
                        elem_id="inference_temperature"
                    )

                    with gr.Row(elem_classes="inference_options_group"):
                        top_p = gr.Slider(
                            minimum=0, maximum=1, value=0.75, step=0.01,
                            label="Top P",
                            elem_id="inference_top_p"
                        )

                        top_k = gr.Slider(
                            minimum=0, maximum=100, value=40, step=1,
                            label="Top K",
                            elem_id="inference_top_k"
                        )

                    num_beams = gr.Slider(
                        minimum=1, maximum=4, value=1, step=1,
                        label="Beams",
                        elem_id="inference_beams"
                    )

                    repetition_penalty = gr.Slider(
                        minimum=0, maximum=2.5, value=1.2, step=0.01,
                        label="Repetition Penalty",
                        elem_id="inference_repetition_penalty"
                    )

                    max_new_tokens = gr.Slider(
                        minimum=0, maximum=4096, value=128, step=1,
                        label="Max New Tokens",
                        elem_id="inference_max_new_tokens"
                    )

                    with gr.Row(elem_id="inference_options_bottom_group"):
                        stream_output = gr.Checkbox(
                            label="Stream Output",
                            elem_id="inference_stream_output",
                            value=True
                        )
                        show_raw = gr.Checkbox(
                            label="Show Raw",
                            elem_id="inference_show_raw",
                            value=default_show_raw
                        )

                with gr.Column():
                    with gr.Row():
                        generate_btn = gr.Button(
                            "Generate", variant="primary", label="Generate", elem_id="inference_generate_btn",
                        )
                        stop_btn = gr.Button(
                            "Stop", variant="stop", label="Stop Iterating", elem_id="inference_stop_btn")

            with gr.Column(elem_id="inference_output_group_container"):
                with gr.Column(elem_id="inference_output_group"):
                    inference_output = gr.Textbox(
                        lines=12, label="Output", elem_id="inference_output")
                    inference_output.style(show_copy_button=True)
                    with gr.Accordion(
                            "Raw Output",
                            open=not default_show_raw,
                            visible=default_show_raw,
                            elem_id="inference_inference_raw_output_accordion"
                    ) as raw_output_group:
                        inference_raw_output = gr.Code(
                            label="Raw Output",
                            show_label=False,
                            language="json",
                            interactive=False,
                            elem_id="inference_raw_output")

        show_raw_change_event = show_raw.change(
            fn=lambda show_raw: gr.Accordion.update(visible=show_raw),
            inputs=[show_raw],
            outputs=[raw_output_group])
        things_that_might_timeout.append(show_raw_change_event)

        reload_selections_event = reload_selections_button.click(
            reload_selections,
            inputs=[lora_model, prompt_template],
            outputs=[lora_model, prompt_template],
        )
        things_that_might_timeout.append(reload_selections_event)

        prompt_template_change_event = prompt_template.change(fn=handle_prompt_template_change, inputs=[prompt_template], outputs=[
            variable_0, variable_1, variable_2, variable_3, variable_4, variable_5, variable_6, variable_7])
        things_that_might_timeout.append(prompt_template_change_event)

        generate_event = generate_btn.click(
            fn=do_inference,
            inputs=[
                lora_model,
                prompt_template,
                variable_0, variable_1, variable_2, variable_3,
                variable_4, variable_5, variable_6, variable_7,
                temperature,
                top_p,
                top_k,
                num_beams,
                repetition_penalty,
                max_new_tokens,
                stream_output,
                show_raw,
            ],
            outputs=[inference_output, inference_raw_output],
            api_name="inference"
        )
        stop_btn.click(fn=None, inputs=None, outputs=None,
                       cancels=[generate_event])

        update_prompt_preview_event = update_prompt_preview_btn.click(fn=update_prompt_preview, inputs=[prompt_template,
                                                                                                        variable_0, variable_1, variable_2, variable_3,
                                                                                                        variable_4, variable_5, variable_6, variable_7,], outputs=preview_prompt)
        things_that_might_timeout.append(update_prompt_preview_event)

        stop_timeoutable_btn = gr.Button(
            "stop not-responding elements",
            elem_id="inference_stop_timeoutable_btn",
            elem_classes="foot_stop_timeoutable_btn")
        stop_timeoutable_btn.click(
            fn=None, inputs=None, outputs=None, cancels=things_that_might_timeout)

    inference_ui_blocks.load(_js="""
    function inference_ui_blocks_js() {
      // Auto load options
      setTimeout(function () {
        document.getElementById('inference_reload_selections_button').click();

        // Workaround default value not shown.
        document.querySelector('#inference_lora_model input').value =
          'tloen/alpaca-lora-7b';
      }, 100);

      // Add tooltips
      setTimeout(function () {
        tippy('#inference_lora_model', {
          placement: 'bottom-start',
          delay: [500, 0],
          animation: 'scale-subtle',
          content:
            'Select a LoRA model form your data directory, or type in a model name on HF (e.g.: <code>tloen/alpaca-lora-7b</code>).',
          allowHTML: true,
        });

        tippy('#inference_prompt_template', {
          placement: 'bottom-start',
          delay: [500, 0],
          animation: 'scale-subtle',
          content:
            'Templates are loaded from the "templates" folder of your data directory. Be sure to select the template that matches your selected LoRA model to get the best results.',
        });

        tippy('#inference_reload_selections_button', {
          placement: 'bottom-end',
          delay: [500, 0],
          animation: 'scale-subtle',
          content: 'Press to reload LoRA Model and Prompt Template selections.',
        });

        document
          .querySelector('#inference_preview_prompt_container .label-wrap')
          .addEventListener('click', function () {
            tippy('#inference_preview_prompt', {
              placement: 'right',
              delay: [500, 0],
              animation: 'scale-subtle',
              content: 'This is the prompt that will be sent to the language model.',
            });

            const update_btn = document.getElementById(
              'inference_update_prompt_preview_btn'
            );
            if (update_btn) update_btn.click();
          });

        function setTooltipForOptions() {
          tippy('#inference_temperature', {
            placement: 'right',
            delay: [500, 0],
            animation: 'scale-subtle',
            content:
              'Controls randomness: Lowering results in less random completions. Higher values (e.g., 1.0) make the model generate more diverse and random outputs. As the temperature approaches zero, the model will become deterministic and repetitive.',
          });

          tippy('#inference_top_p', {
            placement: 'right',
            delay: [500, 0],
            animation: 'scale-subtle',
            content:
              'Controls diversity via nucleus sampling: only the tokens whose cumulative probability exceeds "top_p" are considered. 0.5 means half of all likelihood-weighted options are considered.',
          });

          tippy('#inference_top_k', {
            placement: 'right',
            delay: [500, 0],
            animation: 'scale-subtle',
            content:
              'Controls diversity of the generated text by only considering the "top_k" tokens with the highest probabilities. This method can lead to more focused and coherent outputs by reducing the impact of low probability tokens.',
          });

          tippy('#inference_beams', {
            placement: 'right',
            delay: [500, 0],
            animation: 'scale-subtle',
            content:
              'Number of candidate sequences explored in parallel during text generation using beam search. A higher value increases the chances of finding high-quality, coherent output, but may slow down the generation process.',
          });

          tippy('#inference_repetition_penalty', {
            placement: 'right',
            delay: [500, 0],
            animation: 'scale-subtle',
            content:
              'Applies a penalty to the probability of tokens that have already been generated, discouraging the model from repeating the same words or phrases. The penalty is applied by dividing the token probability by a factor based on the number of times the token has appeared in the generated text.',
          });

          tippy('#inference_max_new_tokens', {
            placement: 'right',
            delay: [500, 0],
            animation: 'scale-subtle',
            content:
              'Limits the maximum number of tokens generated in a single iteration.',
          });

          tippy('#inference_stream_output', {
            placement: 'right',
            delay: [500, 0],
            animation: 'scale-subtle',
            content:
              'When enabled, generated text will be displayed in real-time as it is being produced by the model, allowing you to observe the text generation process as it unfolds.',
          });
        }
        setTooltipForOptions();

        const inference_options_accordion_toggle = document.querySelector(
          '#inference_options_accordion .label-wrap'
        );
        if (inference_options_accordion_toggle) {
          inference_options_accordion_toggle.addEventListener('click', function () {
            setTooltipForOptions();
          });
        }
      }, 100);

      // Show/hide generate and stop button base on the state.
      setTimeout(function () {
        // Make the '#inference_output > .wrap' element appear
        document.getElementById('inference_stop_btn').click();

        setTimeout(function () {
          const output_wrap_element = document.querySelector(
            '#inference_output > .wrap'
          );
          function handle_output_wrap_element_class_change() {
            if (Array.from(output_wrap_element.classList).includes('hide')) {
              document.getElementById('inference_generate_btn').style.display =
                'block';
              document.getElementById('inference_stop_btn').style.display = 'none';
            } else {
              document.getElementById('inference_generate_btn').style.display =
                'none';
              document.getElementById('inference_stop_btn').style.display = 'block';
            }
          }
          new MutationObserver(function (mutationsList, observer) {
            handle_output_wrap_element_class_change();
          }).observe(output_wrap_element, {
            attributes: true,
            attributeFilter: ['class'],
          });
          handle_output_wrap_element_class_change();
        }, 500);
      }, 0);

      // Debounced updating the prompt preview.
      setTimeout(function () {
        function debounce(func, wait) {
          let timeout;
          return function (...args) {
            const context = this;
            clearTimeout(timeout);
            timeout = setTimeout(() => {
              func.apply(context, args);
            }, wait);
          };
        }

        function update_preview() {
          const update_btn = document.getElementById(
            'inference_update_prompt_preview_btn'
          );
          if (!update_btn) return;

          update_btn.click();
        }

        for (let i = 0; i < 8; i++) {
          const e = document.querySelector(`#inference_variable_${i} textarea`);
          if (!e) return;
          e.addEventListener('input', debounce(update_preview, 500));
        }

        const prompt_template_selector = document.querySelector(
          '#inference_prompt_template .wrap-inner'
        );

        if (prompt_template_selector) {
          new MutationObserver(
            debounce(function () {
              if (prompt_template_selector.classList.contains('showOptions')) return;
              update_preview();
            }, 500)
          ).observe(prompt_template_selector, {
            attributes: true,
            attributeFilter: ['class'],
          });
        }
      }, 100);
    }
    """)
