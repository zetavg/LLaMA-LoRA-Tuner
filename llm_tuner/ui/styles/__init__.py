from ...utils.relative_read_file import relative_read_file
from ..css_styles import get_css_styles, register_css_style

register_css_style(
    'accordion',
    relative_read_file(__file__, 'accordion.css')
)
register_css_style(
    'block-reload-btn',
    relative_read_file(__file__, 'block-reload-btn.css')
)
register_css_style(
    'examples-with-width-limit',
    relative_read_file(__file__, 'examples-with-width-limit.css')
)
register_css_style(
    'form-box',
    relative_read_file(__file__, 'form-box.css')
)
register_css_style(
    'json-code-block',
    relative_read_file(__file__, 'json-code-block.css')
)
register_css_style(
    'panel-with-textbox-and-btn',
    relative_read_file(__file__, 'panel-with-textbox-and-btn.css')
)
register_css_style(
    'stop-non-responding-elements-btn',
    relative_read_file(__file__, 'stop-non-responding-elements-btn.css')
)
register_css_style(
    'tag',
    relative_read_file(__file__, 'tag.css')
)
register_css_style(
    'tippy',
    relative_read_file(__file__, 'tippy.css')
)
register_css_style(
    'utility-classes',
    relative_read_file(__file__, 'utility-classes.css')
)
