import time
import json
from textwrap import dedent


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
    message = dedent(f"""
        Hi, I’m currently in UI development mode and do not have access to resources to process your request. However, this behavior is similar to what will actually happen, so you can try and see how it will work!
        """).strip()

    if generation_config.do_sample:
        message += '\n\n'
        message += dedent(f"""
            Generation config:

            ```json
            """).strip()
        message += '\n'
        message += \
            json.dumps(generation_config.to_dict(),
                       ensure_ascii=False, indent=2)
        message += '\n'
        message += dedent(f"""
            ```
            """).strip()

    message += '\n\n'
    message += dedent(f"""
        The following is your prompt:
        """).strip()

    message += "\n"
    message += prompt

    if stream_output:
        def word_generator(message):
            lines = message.split('\n')
            out = ""
            for line in lines:
                words = line.split(' ')
                for i in range(len(words)):
                    if out:
                        out += ' '
                    out += words[i]
                    yield out
                out += "\n"
                yield out

        full_output = ""
        for partial_sentence in word_generator(message):
            full_output = partial_sentence
            decoded_output = partial_sentence
            output = tokenizer(decoded_output)['input_ids']
            yield decoded_output, output, False
            time.sleep(0.05)

        decoded_output = full_output
        output = tokenizer(decoded_output)['input_ids']
        yield decoded_output, output, True

        return
    time.sleep(1)
    decoded_output = message
    output = tokenizer(message)['input_ids']
    yield decoded_output, output, True
    return
