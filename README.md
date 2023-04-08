# 🦙🎛️ LLaMA-LoRA

<a href="https://colab.research.google.com/github/zetavg/LLaMA-LoRA/blob/main/LLaMA_LoRA.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Making evaluating and fine-tuning LLaMA models with low-rank adaptation (LoRA) easy.


## Features

**[See a demo on Hugging Face](https://huggingface.co/spaces/zetavg/LLaMA-LoRA-UI-Demo)** **Only serves UI demonstration. To try training or text generation, [run on Colab](#run-on-google-colab).*

* **[1-click up and running in Google Colab](#run-on-google-colab)** with a standard GPU runtime.
  * Loads and stores data in Google Drive.
* Evaluate various LLaMA LoRA models stored in your folder or from Hugging Face.<br /><a href="https://youtu.be/A3kb4VkDWyY"><img width="640px" src="https://user-images.githubusercontent.com/3784687/230272844-09f7a35b-46bf-4101-b15d-4ddf243b8bef.gif" /></a>
* Fine-tune LLaMA models with different prompt templates and training dataset format.<br /><a href="https://youtu.be/5Db9U8PsaUk"><img width="640px" src="https://user-images.githubusercontent.com/3784687/230277315-9a91d983-1690-4594-9d54-912eda8963ee.gif" /></a>
  * Load JSON and JSONL datasets from your folder, or even paste plain text directly into the UI.
  * Supports Stanford Alpaca [seed_tasks](https://github.com/tatsu-lab/stanford_alpaca/blob/main/seed_tasks.jsonl), [alpaca_data](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json) and [OpenAI "prompt"-"completion"](https://platform.openai.com/docs/guides/fine-tuning/data-formatting) format.


## How to Start

There are various ways to run this app:

* **[Run on Google Colab](#run-on-google-colab)**: The simplest way to get started, all you need is a Google account. Standard (free) GPU runtime is sufficient to run generation and training with micro batch size of 8. However, the text generation and training is much slower than on other cloud services, and Colab might terminate the execution in inactivity while running long tasks.
* **[Run on a cloud service via SkyPilot](#run-on-a-cloud-service-via-skypilot)**: If you have a cloud service (Lambda Labs, GCP, AWS, or Azure) account, you can use SkyPilot to run the app on a cloud service. A cloud bucket can be mounted to preserve your data.
* **[Run locally](#run-locally)**: Depends on the hardware you have.

### Run On Google Colab

Open [this Colab Notebook](https://colab.research.google.com/github/zetavg/LLaMA-LoRA/blob/main/LLaMA_LoRA.ipynb) and select **Runtime > Run All** (`⌘/Ctrl+F9`).

You will be prompted to authorize Google Drive access, as Google Drive will be used to store your data. See the "Config"/"Google Drive" section for settings and more info.

After approximately 5 minutes of running, you will see the public URL in the output of the "Launch"/"Start Gradio UI 🚀" section (like `Running on public URL: https://xxxx.gradio.live`). Open the URL in your browser to use the app.

### Run on a cloud service via SkyPilot

After following the [installation guide of SkyPilot](https://skypilot.readthedocs.io/en/latest/getting-started/installation.html), create a `.yaml` to define a task for running the app:

```yaml
# llama-lora-multitool.yaml

resources:
  accelerators: A10:1  # 1x NVIDIA A10 GPU
  cloud: lambda  # Optional; if left out, SkyPilot will automatically pick the cheapest cloud.

file_mounts:
  # Mount a presisted cloud storage that will be used as the data directory.
  # (to store train datasets trained models)
  # See https://skypilot.readthedocs.io/en/latest/reference/storage.html for details.
  /data:
    name: llama-lora-multitool-data  # Make sure this name is unique or you own this bucket. If it does not exists, SkyPilot will try to create a bucket with this name.
    store: gcs  # Could be either of [s3, gcs]
    mode: MOUNT

# Clone the LLaMA-LoRA repo and install its dependencies.
setup: |
  git clone https://github.com/zetavg/LLaMA-LoRA.git llama_lora
  cd llama_lora && pip install -r requirements.lock.txt
  cd ..
  echo 'Dependencies installed.'

# Start the app.
run: |
  echo 'Starting...'
  python llama_lora/app.py --data_dir='/data' --base_model='decapoda-research/llama-7b-hf' --share
```

Then launch a cluster to run the task:

```
sky launch -c llama-lora-multitool llama-lora-multitool.yaml
```

`-c ...` is an optional flag to specify a cluster name. If not specified, SkyPilot will automatically generate one.

You will see the public URL of the app in the terminal. Open the URL in your browser to use the app.

Note that exiting `sky launch` will only exit log streaming and will not stop the task. You can use `sky queue --skip-finished` to see the status of running or pending tasks, `sky logs <cluster_name> <job_id>` connect back to log streaming, and `sky cancel <cluster_name> <job_id>` to stop a task.

When you are done, run `sky stop <cluster_name>` to stop the cluster. To terminate a cluster instead, run `sky down <cluster_name>`.

### Run locally

<details>
  <summary>Prepare environment with conda</summary>

  ```bash
  conda create -y python=3.8 -n llama-lora-multitool
  conda activate llama-lora-multitool
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


## Acknowledgements

* https://github.com/tloen/alpaca-lora
* https://github.com/lxe/simple-llama-finetuner
* ...

TBC
