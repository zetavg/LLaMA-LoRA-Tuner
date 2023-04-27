from typing import List

css_styles: List[str] = []


def get_css_styles():
    global css_styles
    return "\n".join(css_styles)


def register_css_style(name, style):
    global css_styles
    css_styles.append(style)
