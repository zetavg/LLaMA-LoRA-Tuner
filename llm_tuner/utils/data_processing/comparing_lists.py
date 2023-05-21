import unicodedata
from itertools import zip_longest
from termcolor import colored


def get_width(text):
    if not isinstance(text, str):
        text = str(text)
    return sum(2 if unicodedata.east_asian_width(c) in 'FWA' else 1
               for c in text)


def pad_right(text, width):
    current_width = get_width(text)
    pad_size = width - current_width
    return text + (' ' * pad_size)


def pad_to_same_width(texts):
    width = max([get_width(text) for text in texts])
    return [pad_right(text, width) for text in texts]


def comparing_lists(
    lists,
    labels=None,
    color_labels=False,
    colors=None,
    max_width=80,
    add_blank_line=False,
):
    output = ''

    line_buffers = []

    label_width = 0
    if labels:
        labels = pad_to_same_width(labels)
        label_width = get_width(labels[0]) + 1

    def flush_lines():
        nonlocal output, line_buffers
        if not line_buffers:
            return

        if output and add_blank_line:
            output += '\n'

        for i, line_buffer in enumerate(line_buffers):
            if labels:
                if color_labels:
                    output += colored(
                        labels[i], 'dark_grey',
                        force_color=True) + ' '
                else:
                    output += labels[i]
            color = None
            if colors:
                color = colors[i]
            if color:
                output += colored(line_buffer, color, force_color=True) + '\n'
            else:
                output += line_buffer + '\n'
        line_buffers = []

    def add_items(items):
        items = [str(item) if not isinstance(item, str) else item
                 for item in items]
        items = [item.replace('\n', '\\n')
                 for item in items]
        items = pad_to_same_width(items)

        nonlocal line_buffers, max_width

        if (
            line_buffers and
            max_width and
            (label_width +
             get_width(line_buffers[0]) + get_width(items[0])) > max_width
        ):
            flush_lines()

        for i, item in enumerate(items):
            if i >= len(line_buffers):
                line_buffers.append('')
            line_buffers[i] += ' ' + item

    for items in zip_longest(*lists, fillvalue=''):
        add_items(items)

    flush_lines()

    return output
