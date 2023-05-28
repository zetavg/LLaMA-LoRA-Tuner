from transformers import AutoModelForCausalLM


def download_model(model_name, args):
    try:
        print(f'Downloading model "{model_name}" ...')
        AutoModelForCausalLM.from_pretrained(
            model_name,
            **{
                **args,
                'torch_dtype':
                'make_model_loading_fail_because_we_just_want_to_download'
            }
        )
    except Exception as e:
        if 'make_model_loading_fail_because_we_just_want_to_download' in str(e):
            print(f'Model "{model_name}" downloaded.')
            return
        raise e
