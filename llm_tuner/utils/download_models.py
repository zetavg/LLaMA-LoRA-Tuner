from ..data import get_model_presets


def download_models(only=None):
    model_presets = get_model_presets()

    for mp in model_presets:
        if only:
            has_match = False

            if not isinstance(only, list):
                only = [only]

            for o in only:
                if o in mp.name or o in mp.model_name_or_path:
                    has_match = True
                    break

            if not has_match:
                print(f"Skipping model '{mp.model_name_or_path}' for '{mp.name}': no match against only={only}.")
                continue

        print(f"Preparing model '{mp.model_name_or_path}' for '{mp.name}'...")
        try:
            mp.download_and_cache_model()

        except Exception as e:
            print(f"Error while preparing model '{mp.model_name_or_path}' for '{mp.name}': {e}")
        print()
