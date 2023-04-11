import torch
import transformers

from .streaming_generation_utils import Iteratorize, Stream


def generate(
    # model
    model,
    tokenizer,
    # input
    prompt,
    generation_config,
    max_new_tokens,
    stopping_criteria=[],
    # output options
    stream_output=False
):
    device = get_device()

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generate_params = {
        "input_ids": input_ids,
        "generation_config": generation_config,
        "return_dict_in_generate": True,
        "output_scores": True,
        "max_new_tokens": max_new_tokens,
        "stopping_criteria": transformers.StoppingCriteriaList() + stopping_criteria
    }

    if stream_output:
        # Stream the reply 1 token at a time.
        # This is based on the trick of using 'stopping_criteria' to create an iterator,
        # from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.

        def generate_with_callback(callback=None, **kwargs):
            kwargs["stopping_criteria"].insert(
                0,
                Stream(callback_func=callback)
            )
            with torch.no_grad():
                model.generate(**kwargs)

        def generate_with_streaming(**kwargs):
            return Iteratorize(
                generate_with_callback, kwargs, callback=None
            )

        with generate_with_streaming(**generate_params) as generator:
            for output in generator:
                decoded_output = tokenizer.decode(output, skip_special_tokens=True)
                yield decoded_output, output
                if output[-1] in [tokenizer.eos_token_id]:
                    break
        return  # early return for stream_output

    # Without streaming
    with torch.no_grad():
        generation_output = model.generate(**generate_params)
    output = generation_output.sequences[0]
    decoded_output = tokenizer.decode(output, skip_special_tokens=True)
    yield decoded_output, output
    return


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

    try:
        if torch.backends.mps.is_available():
            return "mps"
    except:  # noqa: E722
        pass
