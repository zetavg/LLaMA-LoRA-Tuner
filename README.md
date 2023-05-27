# 🦙🎛️ LLaMA-LoRA Tuner

<a href="https://colab.research.google.com/github/zetavg/LLaMA-LoRA-Tuner/blob/main/LLaMA_LoRA.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Making evaluating and fine-tuning LLaMA models with low-rank adaptation (LoRA) easy.

> **Update**:
> 
> On the `dev` branch, there's a new Chat UI and a new *Demo Mode* config as a simple and easy way to demonstrate new models. 
> 
> However, the new version does not have the fine-tuning feature yet and is not backward compatible as it uses a new way to define how models are loaded, and also a new format of prompt templates (from [LangChain](https://github.com/hwchase17/langchain)).
>
> For more info, see: https://github.com/zetavg/LLaMA-LoRA-Tuner/discussions/28.
> 
> https://github.com/zetavg/LLaMA-LoRA-Tuner/assets/3784687/ae81a5ed-fe8b-4b17-bea1-455837c2e909

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
# llm-tuner.yaml

resources:
  accelerators: A10:1  # 1x NVIDIA A10 GPU, about US$ 0.6 / hr on Lambda Cloud. Run `sky show-gpus` for supported GPU types, and `sky show-gpus [GPU_NAME]` for the detailed information of a GPU type.
  cloud: lambda  # Optional; if left out, SkyPilot will automatically pick the cheapest cloud.

file_mounts:
  # Mount a presisted cloud storage that will be used as the data directory.
  # (to store train datasets trained models)
  # See https://skypilot.readthedocs.io/en/latest/reference/storage.html for details.
  /data:
    name: llm-tuner-data  # Make sure this name is unique or you own this bucket. If it does not exists, SkyPilot will try to create a bucket with this name.
    store: s3  # Could be either of [s3, gcs]
    mode: MOUNT

# Clone the LLaMA-LoRA Tuner repo and install its dependencies.
setup: |
  conda create -q python=3.8 -n llm-tuner -y
  conda activate llm-tuner

  # Clone the LLaMA-LoRA Tuner repo and install its dependencies
  [ ! -d llm_tuner ] && git clone https://github.com/zetavg/LLaMA-LoRA-Tuner.git llm_tuner
  echo 'Installing dependencies...'
  pip install -r llm_tuner/requirements.lock.txt

  # Optional: install wandb to enable logging to Weights & Biases
  pip install wandb

  # Optional: patch bitsandbytes to workaround error "libbitsandbytes_cpu.so: undefined symbol: cget_col_row_stats"
  BITSANDBYTES_LOCATION="$(pip show bitsandbytes | grep 'Location' | awk '{print $2}')/bitsandbytes"
  [ -f "$BITSANDBYTES_LOCATION/libbitsandbytes_cpu.so" ] && [ ! -f "$BITSANDBYTES_LOCATION/libbitsandbytes_cpu.so.bak" ] && [ -f "$BITSANDBYTES_LOCATION/libbitsandbytes_cuda121.so" ] && echo 'Patching bitsandbytes for GPU support...' && mv "$BITSANDBYTES_LOCATION/libbitsandbytes_cpu.so" "$BITSANDBYTES_LOCATION/libbitsandbytes_cpu.so.bak" && cp "$BITSANDBYTES_LOCATION/libbitsandbytes_cuda121.so" "$BITSANDBYTES_LOCATION/libbitsandbytes_cpu.so"
  conda install -q cudatoolkit -y

  echo 'Dependencies installed.'

  # Optional: Install and setup Cloudflare Tunnel to expose the app to the internet with a custom domain name
  [ -f /data/secrets/cloudflared_tunnel_token.txt ] && echo "Installing Cloudflare" && curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb && sudo dpkg -i cloudflared.deb && sudo cloudflared service uninstall || : && sudo cloudflared service install "$(cat /data/secrets/cloudflared_tunnel_token.txt | tr -d '\n')"

  # Optional: pre-download models
  echo "Pre-downloading base models so that you won't have to wait for long once the app is ready..."
  python llm_tuner/download_base_model.py --base_model_names='decapoda-research/llama-7b-hf,nomic-ai/gpt4all-j'

# Start the app. `hf_access_token`, `wandb_api_key` and `wandb_project` are optional.
run: |
  conda activate llm-tuner
  python llm_tuner/app.py \
    --data_dir='/data' \
    --hf_access_token="$([ -f /data/secrets/hf_access_token.txt ] && cat /data/secrets/hf_access_token.txt | tr -d '\n')" \
    --wandb_api_key="$([ -f /data/secrets/wandb_api_key.txt ] && cat /data/secrets/wandb_api_key.txt | tr -d '\n')" \
    --wandb_project='llm-tuner' \
    --timezone='Atlantic/Reykjavik' \
    --base_model='decapoda-research/llama-7b-hf' \
    --base_model_choices='decapoda-research/llama-7b-hf,nomic-ai/gpt4all-j,databricks/dolly-v2-7b' \
    --share
```

Then launch a cluster to run the task:

```
sky launch -c llm-tuner llm-tuner.yaml
```

`-c ...` is an optional flag to specify a cluster name. If not specified, SkyPilot will automatically generate one.

You will see the public URL of the app in the terminal. Open the URL in your browser to use the app.

Note that exiting `sky launch` will only exit log streaming and will not stop the task. You can use `sky queue --skip-finished` to see the status of running or pending tasks, `sky logs <cluster_name> <job_id>` connect back to log streaming, and `sky cancel <cluster_name> <job_id>` to stop a task.

When you are done, run `sky stop <cluster_name>` to stop the cluster. To terminate a cluster instead, run `sky down <cluster_name>`.

**Remember to stop or shutdown the cluster when you are done to avoid incurring unexpected charges.** Run `sky cost-report` to see the cost of your clusters.

<details>
  <summary>Log into the cloud machine or mount the filesystem of the cloud machine on your local computer</summary>

  To log into the cloud machine, run `ssh <cluster_name>`, such as `ssh llm-tuner`.

  If you have `sshfs` installed on your local machine, you can mount the filesystem of the cloud machine on your local computer by running a command like the following:

  ```bash
  mkdir -p /tmp/llm_tuner_server && umount /tmp/llm_tuner_server || : && sshfs llm-tuner:/ /tmp/llm_tuner_server
  ```
</details>

### Run locally

<details>
  <summary>Prepare environment with conda</summary>

  ```bash
  conda create -y python=3.8 -n llm-tuner
  conda activate llm-tuner
  ```
</details>

```bash
pip install -r requirements.lock.txt
python app.py --data_dir='./data' --base_model='decapoda-research/llama-7b-hf' --timezone='Atlantic/Reykjavik' --share
```

You will see the local and public URLs of the app in the terminal. Open the URL in your browser to use the app.

For more options, see `python app.py --help`.

<details>
  <summary>UI development mode</summary>

  To test the UI without loading the language model, use the `--ui_dev_mode` flag:

  ```bash
  python app.py --data_dir='./data' --base_model='decapoda-research/llama-7b-hf' --share --ui_dev_mode
  ```

  > To use [Gradio Auto-Reloading](https://gradio.app/developing-faster-with-reload-mode/#python-ide-reload), a `config.yaml` file is required since command line arguments are not supported. There's a sample file to start with: `cp config.yaml.sample config.yaml`. Then, just run `gradio app.py`.
</details>


## Usage

See [video on YouTube](https://youtu.be/IoEMgouZ5xU).


## Acknowledgements

* https://github.com/tloen/alpaca-lora
* https://github.com/lxe/simple-llama-finetuner
* ...

TBC
