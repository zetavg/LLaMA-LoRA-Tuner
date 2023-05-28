import fire

from huggingface_hub import snapshot_download


def main(
    base_model_names: str = "",
):
    '''
    Download and cache base models form Hugging Face.

    :param base_model_names: Names of the base model you want to download, seperated by ",". For example: 'decapoda-research/llama-7b-hf,nomic-ai/gpt4all-j'.
    '''

    assert (
        base_model_names
    ), "Please specify --base_model_names, e.g. --base_model_names='decapoda-research/llama-7b-hf,nomic-ai/gpt4all-j'"

    base_model_names_list = base_model_names.split(',')
    base_model_names_list = [name.strip() for name in base_model_names_list]

    print(f"Base models: {', '.join(base_model_names_list)}.")

    for name in base_model_names_list:
        print(f"Preparing {name}...")
        snapshot_download(name)

    print("")
    print("Done.")


if __name__ == "__main__":
    fire.Fire(main)
