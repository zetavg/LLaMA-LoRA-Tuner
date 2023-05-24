function inference_ui_blocks_js() {
  // Auto load options
  setTimeout(function () {
    document.getElementById('inference_reload_selections_button').click();

    // Workaround default value not shown.
    // document.querySelector('#inference_lora_model input').value =
    //   'None';
  }, 50);

  setTimeout(function () {
    document.getElementById('inference_reload_prompt_examples_button').click();
  }, 150);

  // Markdown rendering
//   setTimeout(function () {
//     var textarea = document.querySelector('#inference_output textarea');
//
//     function throttle(func, limit) {
//       var lastRan;
//       var timer;
//       return function() {
//         const context = this;
//         const args = arguments;
//         function runFunc() {
//           lastRan = Date.now();
//           func.apply(context, args);
//         }
//
//         if (timer) clearTimeout(timer);
//         if (!lastRan || (Date.now() - lastRan) > limit) {
//           runFunc();
//         } else {
//           timer = setTimeout(runFunc, limit - (Date.now() - lastRan));
//         }
//       }
//     }
//
//     var throttled_render = throttle(function () {
//       var html = window.markdownit_md.render(textarea.value);
//       document.querySelector('#inference_output_markdown .prose').innerHTML = html;
//     }, 1000);
//
//     var observer = new MutationObserver(function (mutations) {
//       throttled_render();
//     });
//
//     observer.observe(textarea, { attributes: true, childList: true, characterData: true, subtree: true });
//   }, 100);

  // Add tooltips
  setTimeout(function () {
    tippy('#inference_model_preset_select', {
      placement: 'right',
      delay: [500, 0],
      animation: 'scale-subtle',
      content:
        'Select the model to use. Add models by defining new presets on the "Models" / "Preset" tab.',
      allowHTML: true,
    });

    tippy('#inference_prompt_template', {
      placement: 'left',
      delay: [500, 0],
      animation: 'scale-subtle',
      content:
        'Templates are loaded from the "templates" folder of your data directory.<br/><br/>Select the template that matches the data which the model is trained on to get the best results.',
      allowHTML: true,
    });

    tippy('#inference_reload_selections_button', {
      placement: 'top-end',
      delay: [500, 0],
      animation: 'scale-subtle',
      content: 'Press to reload LoRA Model and Prompt Template selections.',
    });

    tippy('#inference_preview_prompt', {
      placement: 'right',
      delay: [500, 0],
      animation: 'scale-subtle',
      content: 'This is the prompt that will be sent to the language model.',
    });

    document
      .querySelector('#inference_preview_prompt_container .label-wrap')
      .addEventListener('click', function () {
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
          '<strong>Controls randomness</strong>: Higher values (e.g., <code>1.0</code>) make the model generate more diverse and random outputs. As the temperature approaches zero, the model will become deterministic and repetitive.</i>',
        allowHTML: true,
      });

      tippy('#inference_top_k', {
        placement: 'right',
        delay: [500, 0],
        animation: 'scale-subtle',
        content:
          '<strong>Controls diversity</strong> of the generated text by only considering the <code>top_k</code> tokens with the highest probabilities.<br /><br />A lower value can lead to more focused and coherent outputs by reducing the impact of tokens with lower probabilities.<br /><br />Will only take effect if Temperature is set to > 0.',
        allowHTML: true,
      });

      tippy('#inference_top_p', {
        placement: 'right',
        delay: [500, 0],
        animation: 'scale-subtle',
        content:
          '<strong>Controls diversity</strong> via nucleus sampling: If set to < 1, only the tokens whose cumulative probability exceeds <code>top_p</code> are considered. <code>0.5</code> means half of all likelihood-weighted options are considered.<br /><br />A lower <code>top_p</code> would lead to less diverse output because a narrower set of potential next tokens is considered.<br /><br />Will only take effect if Temperature is set to > 0.',
        allowHTML: true,
      });

      tippy('#inference_num_beams', {
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

      tippy('#inference_stop_sequence', {
        placement: 'right',
        delay: [500, 0],
        animation: 'scale-subtle',
        content:
          'A sequence where the generation will stop once such sequence is generated. The returned text will not contain the stop sequence. Example: "<code>Human:</code>".',
        allowHTML: true,
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

    // if (inference_options_accordion_toggle) {
    //   inference_options_accordion_toggle.addEventListener('click', function () {
    //     setTooltipForOptions();
    //   });
    // }
  }, 100);

  // Show/hide generate and stop button base on the state.
  setTimeout(function () {
    // Make the '#inference_output > .wrap' element appear
    document.getElementById('inference_init_inference_output_btn').click();

    setTimeout(function () {
      const output_wrap_element = document.querySelector(
        '#inference_output > .wrap'
      );
      var timer1 = null;
      function handle_output_wrap_element_class_change() {
        if (timer1) { clearTimeout(timer1); }
        if (Array.from(output_wrap_element.classList).includes('hide')) {
          var g_btn = document.getElementById('inference_generate_btn');

          // To prevent double click, disable the button for 500ms.
          g_btn.style.pointerEvents = 'none';
          g_btn.style.opacity = 0.5;
          timer1 = setTimeout(function () {
            g_btn.style.pointerEvents = 'auto';
            g_btn.style.opacity = 1;
          }, 500);

          g_btn.style.display = 'block';
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

  // Reload model selection on possible base model change.
  setTimeout(function () {
    const elem = document.getElementById('main_page_tabs_container');
    if (!elem) return;

    let prevClassList = [];

    new MutationObserver(function (mutationsList, observer) {
      const currentPrevClassList = prevClassList;
      const currentClassList = Array.from(elem.classList);
      prevClassList = Array.from(elem.classList);

      if (!currentPrevClassList.includes('hide')) return;
      if (currentClassList.includes('hide')) return;

      const inference_reload_selected_models_btn_elem = document.getElementById('inference_reload_selected_models_btn');

      if (inference_reload_selected_models_btn_elem) inference_reload_selected_models_btn_elem.click();
    }).observe(elem, {
      attributes: true,
      attributeFilter: ['class'],
    });
  }, 0);

  // Debounced updating the prompt preview.
  setTimeout(function () {
    function debounce(func, wait) {
      let timeout;
      return function (...args) {
        const context = this;
        clearTimeout(timeout);
        const fn = () => {
          if (document.querySelector('#inference_preview_prompt > .wrap.default:not(.hide)')) {
            // Preview request is still loading, wait for 10ms and try again.
            timeout = setTimeout(fn, 10);
            return;
          }
          func.apply(context, args);
        };
        timeout = setTimeout(fn, wait);
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

  // [WORKAROUND-UI01]
  setTimeout(function () {
    const inference_output_textarea = document.querySelector(
      '#inference_output textarea'
    );
    if (!inference_output_textarea) return;
    const observer = new MutationObserver(function () {
      if (inference_output_textarea.getAttribute('rows') === '1') {
        setTimeout(function () {
          const inference_generate_btn = document.getElementById(
            'inference_generate_btn'
          );
          if (inference_generate_btn) inference_generate_btn.click();
        }, 10);
      }
    });
    observer.observe(inference_output_textarea, {
      attributes: true,
      attributeFilter: ['rows'],
    });
  }, 100);

  return [];
}
