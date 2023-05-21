from typing import Any, Dict, List, Union

from transformers import PreTrainedTokenizerBase


def tokenize(
    text: str, tokenizer: PreTrainedTokenizerBase
) -> Dict[str, Union[List[int], List[str]]]:
    if not text:
        return {'ids': [], 'tokens': []}
    tokenize_result = tokenizer(text)

    ids: List[int] = tokenize_result['input_ids']  # type: ignore
    tokens = [tokenizer.decode([i]) for i in ids]
    return {
        'ids': ids,
        'tokens': tokens,
    }
