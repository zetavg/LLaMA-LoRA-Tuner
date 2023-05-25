function chat_ui_blocks_js() {

  setTimeout(function () {
    document.getElementById('chat_ui_load_session_btn').click();
  }, 50);

  setInterval(function () {
    document.getElementById('chat_ui_store_session_btn').click();
  }, 1000);

  return []
}
