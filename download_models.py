from typing import Optional, Union

import fire

from llm_tuner.utils.download_models import download_models


def main(
    only: Optional[str] = None,
):
    '''
    Download and cache the model used in model presets.

    :param only: If specified, will only download models with model preset name or model name matching the value, separate multiple values with ",". For example: "--only='llama,gpt4all'".
    '''

    download_only = []

    if isinstance(only, str):
        download_only = only.split(',')
    if isinstance(only, tuple):
        download_only = list(only)

    download_only = [s.strip() for s in download_only]

    if download_only:
        print(
            f"Only downloading models with model preset name or model name matching {download_only}..."
        )
        print()

    download_models(only=download_only)

    print("")
    print("Done.")


if __name__ == "__main__":
    fire.Fire(main)
