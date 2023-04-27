function finetune_ui_blocks_js() {
  // Auto load options
  setTimeout(function () {
    document.getElementById('finetune_reload_selections_button').click();
  }, 100);

  // Add tooltips
  setTimeout(function () {
    tippy('#finetune_reload_selections_button', {
      placement: 'bottom-end',
      delay: [500, 0],
      animation: 'scale-subtle',
      content: 'Press to reload options.',
    });

    tippy('#finetune_template', {
      placement: 'right',
      delay: [500, 0],
      animation: 'scale-subtle',
      content:
        'Select a template for your prompt. <br />To see how the selected template work, select the "Preview" tab and then check "Show actual prompt". <br />Templates are loaded from the "templates" folder of your data directory.',
      allowHTML: true,
    });

    tippy('#finetune_load_dataset_from', {
      placement: 'bottom-start',
      delay: [500, 0],
      animation: 'scale-subtle',
      content:
        '<strong>Text Input</strong>: Paste the dataset directly in the UI.<br/><strong>Data Dir</strong>: Select a dataset in the data directory.',
      allowHTML: true,
    });

    tippy('#finetune_dataset_preview_show_actual_prompt', {
      placement: 'bottom-start',
      delay: [500, 0],
      animation: 'scale-subtle',
      content:
        'Check to show the prompt that will be feed to the language model.',
    });

    tippy('#dataset_plain_text_input_variables_separator', {
      placement: 'bottom',
      delay: [500, 0],
      animation: 'scale-subtle',
      content:
        'Define a separator to separate input variables. Use "\\n" for new lines.',
    });

    tippy('#dataset_plain_text_input_and_output_separator', {
      placement: 'bottom',
      delay: [500, 0],
      animation: 'scale-subtle',
      content:
        'Define a separator to separate the input (prompt) and the output (completion). Use "\\n" for new lines.',
    });

    tippy('#dataset_plain_text_data_separator', {
      placement: 'bottom',
      delay: [500, 0],
      animation: 'scale-subtle',
      content:
        'Define a separator to separate different rows of the train data. Use "\\n" for new lines.',
    });

    tippy('#finetune_dataset_text_load_sample_button', {
      placement: 'bottom-start',
      delay: [500, 0],
      animation: 'scale-subtle',
      content:
        'Press to load a sample dataset of the current selected format into the textbox.',
    });

    tippy('#finetune_evaluate_data_count', {
      placement: 'bottom',
      delay: [500, 0],
      animation: 'scale-subtle',
      content:
        'While setting a value larger than 0, the checkpoint with the lowest loss on the evaluation data will be saved as the final trained model, thereby helping to prevent overfitting.',
    });

    tippy('#finetune_save_total_limit', {
      placement: 'bottom',
      delay: [500, 0],
      animation: 'scale-subtle',
      content:
        'Total amount of checkpoints to preserve. Older checkpoints will be deleted.',
    });
    tippy('#finetune_save_steps', {
      placement: 'bottom',
      delay: [500, 0],
      animation: 'scale-subtle',
      content:
        'Number of updates steps before two checkpoint saves.',
    });
    tippy('#finetune_logging_steps', {
      placement: 'bottom',
      delay: [500, 0],
      animation: 'scale-subtle',
      content:
        'Number of update steps between two logs.',
    });

    tippy('#finetune_model_name', {
      placement: 'bottom',
      delay: [500, 0],
      animation: 'scale-subtle',
      content:
        'The name of the new LoRA model. Must be unique.',
    });

    tippy('#finetune_continue_from_model', {
      placement: 'right',
      delay: [500, 0],
      animation: 'scale-subtle',
      content:
        'Select a LoRA model to train a new model on top of that model. You can also type in a model name on Hugging Face Hub, such as <code>tloen/alpaca-lora-7b</code>.<br /><br />💡 To reload the training parameters of one of your previously trained models, select it here and click the <code>Load training parameters from selected model</code> button, then un-select it.',
      allowHTML: true,
    });

    tippy('#finetune_continue_from_checkpoint', {
      placement: 'right',
      delay: [500, 0],
      animation: 'scale-subtle',
      content:
        'If a checkpoint is selected, training will resume from that specific checkpoint, bypassing any previously completed steps up to the checkpoint\'s moment. <br /><br />💡 Use this option to resume an unfinished training session. Remember to click the <code>Load training parameters from selected model</code> button and select the same dataset for training.',
      allowHTML: true,
    });
  }, 100);

  // Show/hide start and stop button base on the state.
  setTimeout(function () {
    // Make the '#finetune_training_indicator > .wrap' element appear
    // if (!document.querySelector('#finetune_training_indicator > .wrap')) {
    //   document.getElementById('finetune_confirm_stop_btn').click();
    // }

    setTimeout(function () {
      let resetStopButtonTimer;
      document
        .getElementById('finetune_stop_btn')
        .addEventListener('click', function () {
          if (resetStopButtonTimer) clearTimeout(resetStopButtonTimer);
          resetStopButtonTimer = setTimeout(function () {
            document.getElementById('finetune_stop_btn').style.display = 'block';
            document.getElementById('finetune_confirm_stop_btn').style.display =
              'none';
          }, 5000);
          document.getElementById('finetune_confirm_stop_btn').style['pointer-events'] =
            'none';
          setTimeout(function () {
            document.getElementById('finetune_confirm_stop_btn').style['pointer-events'] =
              'inherit';
          }, 300);
          document.getElementById('finetune_stop_btn').style.display = 'none';
          document.getElementById('finetune_confirm_stop_btn').style.display =
            'block';
        });
      // const training_indicator_wrap_element = document.querySelector(
      //   '#finetune_training_indicator > .wrap'
      // );
      const training_indicator_element = document.querySelector(
        '#finetune_training_indicator'
      );
      let isTraining = undefined;
      function handle_training_indicator_change() {
        // const wrapperHidden = Array.from(training_indicator_wrap_element.classList).includes('hide');
        const hidden = Array.from(training_indicator_element.classList).includes('hidden');
        const newIsTraining = !(/* wrapperHidden && */ hidden);
        if (newIsTraining === isTraining) return;
        isTraining = newIsTraining;
        if (!isTraining) {
          if (resetStopButtonTimer) clearTimeout(resetStopButtonTimer);
          document.getElementById('finetune_start_btn').style.display = 'block';
          document.getElementById('finetune_stop_btn').style.display = 'none';
          document.getElementById('finetune_confirm_stop_btn').style.display =
            'none';
        } else {
          document.getElementById('finetune_start_btn').style.display = 'none';
          document.getElementById('finetune_stop_btn').style.display = 'block';
          document.getElementById('finetune_confirm_stop_btn').style.display =
            'none';
        }
      }
      // new MutationObserver(function (mutationsList, observer) {
      //   handle_training_indicator_change();
      // }).observe(training_indicator_wrap_element, {
      //   attributes: true,
      //   attributeFilter: ['class'],
      // });
      new MutationObserver(function (mutationsList, observer) {
        handle_training_indicator_change();
      }).observe(training_indicator_element, {
        attributes: true,
        attributeFilter: ['class'],
      });
      handle_training_indicator_change();
    }, 500);
  }, 0);

  return [];
}
