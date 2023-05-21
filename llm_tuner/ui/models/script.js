function models_ui_js() {
  // Load data
  setTimeout(function () {
    document.getElementById('models_preset_list_reload_button').click();
  }, 100);

  // Tooltips
  setTimeout(function () {
    tippy('#models_ui', {
      placement: 'top',
      delay: [500, 0],
      animation: 'scale-subtle',
      content:
        'Hello <code>world</code>.',
      allowHTML: true,
    });
  }, 100);

  return [];
}
