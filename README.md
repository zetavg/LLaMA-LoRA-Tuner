# 🦙🎛️ LLaMA-LoRA Tuner

<a href="https://colab.research.google.com/github/zetavg/LLaMA-LoRA-Tuner/blob/main/LLaMA_LoRA.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Making evaluating and fine-tuning LLaMA models with low-rank adaptation (LoRA) easy.


## Features

**[See a demo on Hugging Face](https://huggingface.co/spaces/zetavg/LLaMA-LoRA-UI-Demo)** **Only serves UI demonstration. To try training or text generation, [run on Colab](#run-on-google-colab).*

* **[1-click up and running in Google Colab](#run-on-google-colab)** with a standard GPU runtime.
  * Loads and stores data in Google Drive.
* Evaluate various LLaMA LoRA models stored in your folder or from Hugging Face.<br /><a href="https://youtu.be/IoEMgouZ5xU"><img width="640px" src="https://user-images.githubusercontent.com/3784687/231023326-f28c84e2-df74-4179-b0ac-c25c4e8ca001.gif" /></a>
* Switch between base models such as `decapoda-research/llama-7b-hf`, `nomic-ai/gpt4all-j`, `databricks/dolly-v2-7b`, `EleutherAI/gpt-j-6b`, or `EleutherAI/pythia-6.9b`.
* Fine-tune LLaMA models with different prompt templates and training dataset format.<br /><a href="https://youtu.be/IoEMgouZ5xU?t=60"><img width="640px" src="https://user-images.githubusercontent.com/3784687/231026640-b5cf5c79-9fe9-430b-8d4e-7346eb9567ad.gif" /></a>
  * Load JSON and JSONL datasets from your folder, or even paste plain text directly into the UI.
  * Supports Stanford Alpaca [seed_tasks](https://github.com/tatsu-lab/stanford_alpaca/blob/main/seed_tasks.jsonl), [alpaca_data](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json) and [OpenAI "prompt"-"completion"](https://platform.openai.com/docs/guides/fine-tuning/data-formatting) format.
  * Use prompt templates to keep your dataset DRY.


## How to Start

There are various ways to run this app:

* **[Run on Google Colab](#run-on-google-colab)**: The simplest way to get started, all you need is a Google account. Standard (free) GPU runtime is sufficient to run generation and training with micro batch size of 8. However, the text generation and training is much slower than on other cloud services, and Colab might terminate the execution in inactivity while running long tasks.
* **[Run on a cloud service via SkyPilot](#run-on-a-cloud-service-via-skypilot)**: If you have a cloud service (Lambda Labs, GCP, AWS, or Azure) account, you can use SkyPilot to run the app on a cloud service. A cloud bucket can be mounted to preserve your data.
* **[Run locally](#run-locally)**: Depends on the hardware you have.

### Run On Google Colab

*See [video](https://youtu.be/lByYOMdy9h4) for step-by-step instructions.*

Open [this Colab Notebook](https://colab.research.google.com/github/zetavg/LLaMA-LoRA-Tuner/blob/main/LLaMA_LoRA.ipynb) and select **Runtime > Run All** (`⌘/Ctrl+F9`).

You will be prompted to authorize Google Drive access, as Google Drive will be used to store your data. See the "Config"/"Google Drive" section for settings and more info.

After approximately 5 minutes of running, you will see the public URL in the output of the "Launch"/"Start Gradio UI 🚀" section (like `Running on public URL: https://xxxx.gradio.live`). Open the URL in your browser to use the app.

### Run on a cloud service via SkyPilot

After following the [installation guide of SkyPilot](https://skypilot.readthedocs.io/en/latest/getting-started/installation.html), create a `.yaml` to define a task for running the app:

```yaml
# llama-lora-tuner.yaml

resources:
  accelerators: A10:1  # 1x NVIDIA A10 GPU, about US$ 0.6 / hr on Lambda Cloud.
  cloud: lambda  # Optional; if left out, SkyPilot will automatically pick the cheapest cloud.

file_mounts:
  # Mount a presisted cloud storage that will be used as the data directory.
  # (to store train datasets trained models)
  # See https://skypilot.readthedocs.io/en/latest/reference/storage.html for details.
  /data:
    name: llama-lora-tuner-data  # Make sure this name is unique or you own this bucket. If it does not exists, SkyPilot will try to create a bucket with this name.
    store: s3  # Could be either of [s3, gcs]
    mode: MOUNT

# Clone the LLaMA-LoRA Tuner repo and install its dependencies.
setup: |
  git clone https://github.com/zetavg/LLaMA-LoRA-Tuner.git llama_lora_tuner
  cd llama_lora_tuner && pip install -r requirements.lock.txt
  pip install wandb
  cd ..
  echo 'Dependencies installed.'
  echo 'Pre-downloading base models so that you won't have to wait for long once the app is ready...'
  python llama_lora_tuner/download_base_model.py --base_model_names='decapoda-research/llama-7b-hf,nomic-ai/gpt4all-j,databricks/dolly-v2-7b'

# Start the app.
run: |
  echo 'Starting...'
  python llama_lora_tuner/app.py --data_dir='/data' --wandb_api_key="$([ -f /data/secrets/wandb_api_key ] && cat /data/secrets/wandb_api_key | tr -d '\n')" --base_model=decapoda-research/llama-7b-hf --base_model_choices='decapoda-research/llama-7b-hf,nomic-ai/gpt4all-j,databricks/dolly-v2-7b --share
```

Then launch a cluster to run the task:

```
sky launch -c llama-lora-tuner llama-lora-tuner.yaml
```

`-c ...` is an optional flag to specify a cluster name. If not specified, SkyPilot will automatically generate one.

You will see the public URL of the app in the terminal. Open the URL in your browser to use the app.

Note that exiting `sky launch` will only exit log streaming and will not stop the task. You can use `sky queue --skip-finished` to see the status of running or pending tasks, `sky logs <cluster_name> <job_id>` connect back to log streaming, and `sky cancel <cluster_name> <job_id>` to stop a task.

When you are done, run `sky stop <cluster_name>` to stop the cluster. To terminate a cluster instead, run `sky down <cluster_name>`.

### Run locally

<details>
  <summary>Prepare environment with conda</summary>

  ```bash
  conda create -y python=3.8 -n llama-lora-tuner
  conda activate llama-lora-tuner
  ```
</details>

```bash
pip install -r requirements.lock.txt
python app.py --data_dir='./data' --base_model='decapoda-research/llama-7b-hf' --share
```

You will see the local and public URLs of the app in the terminal. Open the URL in your browser to use the app.

For more options, see `python app.py --help`.

<details>
  <summary>UI development mode</summary>

  To test the UI without loading the language model, use the `--ui_dev_mode` flag:

  ```bash
  python app.py --data_dir='./data' --base_model='decapoda-research/llama-7b-hf' --share --ui_dev_mode
  ```
</details>


## Usage

See [video on YouTube](https://youtu.be/IoEMgouZ5xU).


## Acknowledgements

* https://github.com/tloen/alpaca-lora
* https://github.com/lxe/simple-llama-finetuner
* ...

TBC
