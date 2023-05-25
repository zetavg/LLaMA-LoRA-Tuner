function inference_ui_starter_tooltips() {
  setTimeout(function () {
    add_tooltip('#inference_model_preset_select', {
      placement: 'right',
      content:
        'Select the model to use. Add models by defining new presets on the "Models" / "Preset" tab.',
    });

    add_tooltip('#inference_prompt_template_select', {
      placement: 'left',
      content:
        'Templates are loaded from the "templates" folder of your data directory.<br/><br/>Select the template that matches the data which the model is trained on to get the best results.',
    });
  }, 100);

  return [];
}
