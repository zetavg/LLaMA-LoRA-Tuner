def deep_sort_dict(d):
    if isinstance(d, dict):
        return dict(
            sorted(
                ((k, deep_sort_dict(v)) for k, v in d.items()),
                key=lambda x: x[0]
            )
        )
    elif isinstance(d, list):
        return [deep_sort_dict(v) for v in d]
    else:
        return d
