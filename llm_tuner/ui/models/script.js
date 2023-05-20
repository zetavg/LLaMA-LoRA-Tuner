function models_ui_js() {
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
