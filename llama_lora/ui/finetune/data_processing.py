import json
from ...utils.data import get_dataset_content

from .values import (
    default_dataset_plain_text_input_variables_separator,
    default_dataset_plain_text_input_and_output_separator,
    default_dataset_plain_text_data_separator,
)


def get_data_from_input(load_dataset_from, dataset_text, dataset_text_format,
                        dataset_plain_text_input_variables_separator,
                        dataset_plain_text_input_and_output_separator,
                        dataset_plain_text_data_separator,
                        dataset_from_data_dir, prompter):
    if load_dataset_from == "Text Input":
        if dataset_text_format == "JSON":
            data = json.loads(dataset_text)

        elif dataset_text_format == "JSON Lines":
            lines = dataset_text.split('\n')
            data = []
            for i, line in enumerate(lines):
                line_number = i + 1
                try:
                    data.append(json.loads(line))
                except Exception as e:
                    raise ValueError(
                        f"Error parsing JSON on line {line_number}: {e}")

        else:  # Plain Text
            data = parse_plain_text_input(
                dataset_text,
                (
                    dataset_plain_text_input_variables_separator or
                    default_dataset_plain_text_input_variables_separator
                ).replace("\\n", "\n"),
                (
                    dataset_plain_text_input_and_output_separator or
                    default_dataset_plain_text_input_and_output_separator
                ).replace("\\n", "\n"),
                (
                    dataset_plain_text_data_separator or
                    default_dataset_plain_text_data_separator
                ).replace("\\n", "\n"),
                prompter.get_variable_names()
            )

    else:  # Load dataset from data directory
        data = get_dataset_content(dataset_from_data_dir)

    return data


def parse_plain_text_input(
    value,
    variables_separator, input_output_separator, data_separator,
    variable_names
):
    items = value.split(data_separator)
    result = []
    for item in items:
        parts = item.split(input_output_separator)
        variables = get_val_from_arr(parts, 0, "").split(variables_separator)
        variables = [it.strip() for it in variables]
        variables_dict = {name: var for name,
                          var in zip(variable_names, variables)}
        output = get_val_from_arr(parts, 1, "").strip()
        result.append({'variables': variables_dict, 'output': output})
    return result


def get_val_from_arr(arr, index, default=None):
    return arr[index] if -len(arr) <= index < len(arr) else default
