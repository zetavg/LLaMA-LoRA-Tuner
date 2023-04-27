import numpy as np
from typing import List, Any, Iterator


def sample_evenly_it(input_list: List[Any], max_elements: int = 1000) -> Iterator[Any]:
    if len(input_list) <= max_elements:
        yield from input_list
    else:
        step = len(input_list) / max_elements
        indices = np.arange(0, len(input_list), step).astype(int)
        yield from (input_list[i] for i in indices)


def sample_evenly(input_list: List[Any], max_elements: int = 1000) -> List[Any]:
    return list(sample_evenly_it(input_list, max_elements))
