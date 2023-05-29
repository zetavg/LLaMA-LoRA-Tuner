import pdb
from fire.core import completion
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
    # max_new_tokens,
    stopping_criteria=[],
    stop_sequences=[],
    include_stop_sequence_in_returned_text=False,
    skip_special_tokens=True,
    # output options
    stream_output=False
):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)

    len_prompt = len(prompt)

    generate_params = {
        "input_ids": input_ids,
        "generation_config": generation_config,
        "return_dict_in_generate": True,
        "output_scores": True,
        # "max_new_tokens": max_new_tokens,
        "stopping_criteria": transformers.StoppingCriteriaList() + stopping_criteria
    }

    if not isinstance(stop_sequences, list):
        stop_sequences = [stop_sequences]

    if stop_sequences:
        def stop_sequences_stopping_criteria(
            input_ids, score, **kwargs
        ):
            nonlocal tokenizer, stop_sequences, len_prompt
            ids = input_ids[0]
            output = tokenizer.decode(ids)

            new_output = output[len_prompt:]
            for s in stop_sequences:
                # if s == '\n' and len(ids) < 8:
                #     continue
                if s in new_output:
                    return True
            return False
        generate_params['stopping_criteria'].append(
            stop_sequences_stopping_criteria
        )

    if '/dolly' in tokenizer.name_or_path:
        # dolly has additional_special_tokens as ['### End', '### Instruction:', '### Response:'], skipping them will break the prompter's reply extraction.
        skip_special_tokens = False
        # Ensure generation stops once it generates "### End"
        end_key_token_id = tokenizer.encode("### End")
        end_key_token_id = end_key_token_id[0]  # 50277
        if isinstance(generate_params['generation_config'].eos_token_id, str):
            generate_params['generation_config'].eos_token_id = [
                generate_params['generation_config'].eos_token_id]
        elif not generate_params['generation_config'].eos_token_id:
            generate_params['generation_config'].eos_token_id = []
        generate_params['generation_config'].eos_token_id.append(
            end_key_token_id)

    def post_process_returned_text(text, is_streaming=False):
        if not stop_sequences:
            return text
        if include_stop_sequence_in_returned_text:
            return text

        prompt_text = text[:len_prompt]
        completion_text = text[len_prompt:]
        for s in stop_sequences:
            completion_text = completion_text.split(s)[0]
            if is_streaming:
                completion_text = remove_overlap_sp(completion_text, s)

        return (prompt_text + completion_text).rstrip()

    if stream_output:
        # Stream the reply 1 token at a time.
        # This is based on the trick of using 'stopping_criteria' to create an iterator,
        # from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.
        generation_output = None

        def generate_with_callback(callback=None, **kwargs):
            nonlocal generation_output
            kwargs["stopping_criteria"].insert(
                0,
                Stream(callback_func=callback)
            )
            with torch.no_grad():
                generation_output = model.generate(**kwargs)

        def generate_with_streaming(**kwargs):
            return Iteratorize(
                generate_with_callback, kwargs, callback=None
            )

        with generate_with_streaming(**generate_params) as generator:
            for output in generator:
                decoded_output = tokenizer.decode(
                    output, skip_special_tokens=skip_special_tokens)
                yield (
                    post_process_returned_text(
                        decoded_output,
                        is_streaming=True
                    ),
                    output,
                    False
                )

        if generation_output:
            output = generation_output.sequences[0]
            decoded_output = tokenizer.decode(
                output, skip_special_tokens=skip_special_tokens)
            output_as_list = output
            if not isinstance(output_as_list, list):
                output_as_list = output_as_list.tolist()
            yield (
                post_process_returned_text(decoded_output),
                output,
                True
            )

        return  # early return for stream_output

    # Without streaming
    with torch.no_grad():
        generation_output = model.generate(**generate_params)
    output = generation_output.sequences[0]
    decoded_output = tokenizer.decode(
        output, skip_special_tokens=skip_special_tokens)
    output_as_list = output
    if not isinstance(output_as_list, list):
        output_as_list = output_as_list.tolist()
    yield (
        post_process_returned_text(decoded_output),
        output,
        True
    )
    return


def remove_overlap_sp(str1, str2):
    min_len = min(len(str1), len(str2))

    # Find the longest common suffix-prefix
    for i in range(min_len, 0, -1):
        if str1[-i:] == str2[:i]:
            return str1[:-i]

    # No overlap
    return str1
