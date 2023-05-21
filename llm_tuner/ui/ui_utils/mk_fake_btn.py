def mk_fake_btn(fake_btn, elem_id, real_elem_id, delay=5000):
    fake_btn.click(
        fn=None, inputs=None, outputs=None,
        _js=f"""
            function () {{
                const fake_btn = document.getElementById('{elem_id}');
                const real_btn = document.getElementById('{real_elem_id}');
                fake_btn.style.display = 'none';
                real_btn.style.pointerEvents = 'none';
                real_btn.style.opacity = 0.8;
                real_btn.style.display = 'block';
                if (window['fake_btn_{elem_id}_timer_1']) {{
                    clearTimeout(window['fake_btn_{elem_id}_timer_1']);
                }}
                if (window['fake_btn_{elem_id}_timer_2']) {{
                    clearTimeout(window['fake_btn_{elem_id}_timer_2']);
                }}
                window['fake_btn_{elem_id}_timer_1'] = setTimeout(function () {{
                    real_btn.style.pointerEvents = '';
                    real_btn.style.opacity = 1;
                }}, 200);
                var restore = function () {{
                    fake_btn.style.display = '';
                    real_btn.style.display = 'none';
                }}
                window['fake_btn_{elem_id}_timer_2'] = setTimeout(restore, {delay});
                real_btn.addEventListener('click', restore);
                return [];
            }}
        """
    )
