import os
import sys
import importlib
from typing import Any, List

import json

import fire
import torch
import transformers
from datasets import Dataset, load_dataset


from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer


def train(
    # model/data params
    base_model: Any,
    tokenizer: Any,
    output_dir: str,
    train_dataset_data: List[Any],
    # training hyperparams
    micro_batch_size: int = 4,
    gradient_accumulation_steps: int = 32,
    num_train_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000,  # TODO: use percentage
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # either training checkpoint or final adapter
    resume_from_checkpoint: str = None,
    save_steps: int = 200,
    save_total_limit: int = 3,
    logging_steps: int = 10,
    # logging
    callbacks: List[Any] = [],
    # wandb params
    wandb_api_key = None,
    wandb_project: str = "",
    wandb_group = None,
    wandb_run_name: str = "",
    wandb_tags: List[str] = [],
    wandb_watch: str = "false",  # options: false | gradients | all
    wandb_log_model: str = "true",  # options: false | true
):
    # for logging
    finetune_args = {
        'micro_batch_size': micro_batch_size,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'num_train_epochs': num_train_epochs,
        'learning_rate': learning_rate,
        'cutoff_len': cutoff_len,
        'lora_r': lora_r,
        'lora_alpha': lora_alpha,
        'lora_dropout': lora_dropout,
        'lora_target_modules': lora_target_modules,
        'train_on_inputs': train_on_inputs,
        'group_by_length': group_by_length,
        'save_steps': save_steps,
        'save_total_limit': save_total_limit,
        'logging_steps': logging_steps,
    }

    if wandb_api_key:
        os.environ["WANDB_API_KEY"] = wandb_api_key

    # wandb: WARNING Changes to your `wandb` environment variables will be ignored because your `wandb` session has already started. For more information on how to modify your settings with `wandb.init()` arguments, please refer to https://wandb.me/wandb-init.
    # if wandb_project:
    #     os.environ["WANDB_PROJECT"] = wandb_project
    # if wandb_run_name:
    #     os.environ["WANDB_RUN_NAME"] = wandb_run_name
    if wandb_watch:
        os.environ["WANDB_WATCH"] = wandb_watch
    if wandb_log_model:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model
    use_wandb = (wandb_project and len(wandb_project) > 0) or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
        )
    if use_wandb:
        os.environ['WANDB_MODE'] = "online"
        wandb = importlib.import_module("wandb")
        wandb.init(
            project=wandb_project,
            resume="auto",
            group=wandb_group,
            name=wandb_run_name,
            tags=wandb_tags,
            reinit=True,
            magic=True,
            config={'finetune_args': finetune_args},
            # id=None  # used for resuming
            )
    else:
        os.environ['WANDB_MODE'] = "disabled"

    if os.path.exists(output_dir):
        if (not os.path.isdir(output_dir)) or os.path.exists(os.path.join(output_dir, 'adapter_config.json')):
            raise ValueError(
                f"The output directory already exists and is not empty. ({output_dir})")

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    model = base_model
    if isinstance(model, str):
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map=device_map,
        )

    if isinstance(tokenizer, str):
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = data_point["prompt"] + data_point["completion"]
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = data_point["prompt"]
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    # will fail anyway.
    try:
        model = prepare_model_for_int8_training(model)
    except Exception as e:
        print(
            f"Got error while running prepare_model_for_int8_training(model), maybe the model has already be prepared. Original error: {e}.")

    # model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    # If train_dataset_data is a list, convert it to datasets.Dataset
    if isinstance(train_dataset_data, list):
        with open(os.path.join(output_dir, "train_data_samples.json"), 'w') as file:
            json.dump(list(train_dataset_data[:100]), file, indent=2)
        train_dataset_data = Dataset.from_list(train_dataset_data)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    # Be more transparent about the % of trainable params.
    model.print_trainable_parameters()

    if val_set_size > 0:
        train_val = train_dataset_data.train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = train_dataset_data.shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=logging_steps,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=save_steps if val_set_size > 0 else None,
            save_steps=save_steps,
            output_dir=output_dir,
            save_total_limit=save_total_limit,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=callbacks,
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, "trainer_args.json"), 'w') as trainer_args_json_file:
        json.dump(trainer.args.to_dict(), trainer_args_json_file, indent=2)
    with open(os.path.join(output_dir, "finetune_args.json"), 'w') as finetune_args_json_file:
        json.dump(finetune_args, finetune_args_json_file, indent=2)

    # Not working, will only give us ["prompt", "completion", "input_ids", "attention_mask", "labels"]
    # if train_data:
    #     with open(os.path.join(output_dir, "train_dataset_samples.json"), 'w') as file:
    #         json.dump(list(train_data[:100]), file, indent=2)
    # if val_data:
    #     with open(os.path.join(output_dir, "eval_dataset_samples.json"), 'w') as file:
    #         json.dump(list(val_data[:100]), file, indent=2)

    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    train_output = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}.")

    with open(os.path.join(output_dir, "trainer_log_history.jsonl"), 'w') as trainer_log_history_jsonl_file:
        trainer_log_history = "\n".join(
            [json.dumps(line) for line in trainer.state.log_history])
        trainer_log_history_jsonl_file.write(trainer_log_history)

    with open(os.path.join(output_dir, "train_output.json"), 'w') as train_output_json_file:
        json.dump(train_output, train_output_json_file, indent=2)

    return train_output
