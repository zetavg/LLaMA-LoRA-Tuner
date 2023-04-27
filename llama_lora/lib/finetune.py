import os
import sys
import re
import importlib
from typing import Any, List, Union

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
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer


def train(
    # model/data params
    base_model: Any,
    tokenizer: Any,
    output_dir: str,
    train_data: List[Any],
    #
    load_in_8bit=True,
    fp16=True,
    bf16=False,
    gradient_checkpointing=False,
    # training hyperparams
    micro_batch_size: int = 4,
    gradient_accumulation_steps: int = 32,
    num_train_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    lora_modules_to_save: Union[List[str], None] = [],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # either training checkpoint or final adapter
    resume_from_checkpoint=None,
    save_steps: int = 200,
    save_total_limit: int = 3,
    logging_steps: int = 10,
    #
    additional_training_arguments: Union[dict, str, None] = None,
    additional_lora_config: Union[dict, str, None] = None,
    # logging
    callbacks: List[Any] = [],
    # wandb params
    wandb_api_key=None,
    wandb_project: str = "",
    wandb_group=None,
    wandb_run_name: str = "",
    wandb_tags: List[str] = [],
    wandb_watch: str = "false",  # options: false | gradients | all
    wandb_log_model: str = "true",  # options: false | true
    additional_wandb_config: Union[dict, None] = None,
    hf_access_token: Union[str, None] = None,
    status_message_callback: Any = None,
    params_info_callback: Any = None,
):
    if status_message_callback:
        cb_result = status_message_callback("Preparing...")
        if cb_result:
            return

    if lora_modules_to_save is not None and len(lora_modules_to_save) <= 0:
        lora_modules_to_save = None

    if isinstance(additional_training_arguments, str):
        additional_training_arguments = additional_training_arguments.strip()
    if not additional_training_arguments:
        additional_training_arguments = None
    if isinstance(additional_training_arguments, str):
        try:
            additional_training_arguments = json.loads(
                additional_training_arguments)
        except Exception as e:
            raise ValueError(
                f"Could not parse additional_training_arguments: {e}")

    if isinstance(additional_lora_config, str):
        additional_lora_config = additional_lora_config.strip()
    if not additional_lora_config:
        additional_lora_config = None
    if isinstance(additional_lora_config, str):
        try:
            additional_lora_config = json.loads(additional_lora_config)
        except Exception as e:
            raise ValueError(f"Could not parse additional_lora_config: {e}")

    # for logging
    finetune_args = {
        'micro_batch_size': micro_batch_size,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'num_train_epochs': num_train_epochs,
        'learning_rate': learning_rate,
        'cutoff_len': cutoff_len,
        'val_set_size': val_set_size,
        'lora_r': lora_r,
        'lora_alpha': lora_alpha,
        'lora_dropout': lora_dropout,
        'lora_target_modules': lora_target_modules,
        'lora_modules_to_save': lora_modules_to_save or [],
        'train_on_inputs': train_on_inputs,
        'group_by_length': group_by_length,
        'load_in_8bit': load_in_8bit,
        'fp16': fp16,
        'bf16': bf16,
        'gradient_checkpointing': gradient_checkpointing,
        'save_steps': save_steps,
        'save_total_limit': save_total_limit,
        'logging_steps': logging_steps,
        'additional_training_arguments': additional_training_arguments,
        'additional_lora_config': additional_lora_config,
    }
    if val_set_size and val_set_size > 0:
        finetune_args['val_set_size'] = val_set_size
    # if lora_modules_to_save:
    #     finetune_args['lora_modules_to_save'] = lora_modules_to_save
    if resume_from_checkpoint:
        finetune_args['resume_from_checkpoint'] = resume_from_checkpoint

    wandb = None
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
        if additional_wandb_config:
            wandb.config.update(additional_wandb_config)
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

    if status_message_callback:
        if isinstance(base_model, str):
            cb_result = status_message_callback(
                f"Preparing model '{base_model}' for training...")
            if cb_result:
                return
        else:
            cb_result = status_message_callback(
                "Preparing model for training...")
            if cb_result:
                return

    model = base_model
    if isinstance(model, str):
        model_name = model
        print(f"Loading base model {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_in_8bit,
            torch_dtype=torch.float16,
            llm_int8_skip_modules=lora_modules_to_save,
            device_map=device_map,
            use_auth_token=hf_access_token
        )
        if re.match("[^/]+/llama", model_name):
            print(f"Setting special tokens for LLaMA model {model_name}...")
            model.config.pad_token_id = 0
            model.config.bos_token_id = 1
            model.config.eos_token_id = 2

        print(f"Loaded model {model_name}")

    if isinstance(tokenizer, str):
        tokenizer_name = tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer, use_auth_token=hf_access_token
            )
        except Exception as e:
            if 'LLaMATokenizer' in str(e):
                tokenizer = LlamaTokenizer.from_pretrained(
                    tokenizer_name,
                    use_auth_token=hf_access_token
                )
            else:
                raise e

        if re.match("[^/]+/llama", tokenizer_name):
            print(
                f"Setting special tokens for LLaMA tokenizer {tokenizer_name}...")
            tokenizer.pad_token_id = 0
            tokenizer.bos_token_id = 1
            tokenizer.eos_token_id = 2

        print(f"Loaded tokenizer {tokenizer_name}")

    # tokenizer.pad_token_id = (
    #     0  # unk. we want this to be different from the eos token
    # )
    tokenizer.padding_side = "left"  # Allow batched inference

    try:
        model = prepare_model_for_int8_training(model)
    except Exception as e:
        print(
            f"Got error while running prepare_model_for_int8_training(model), maybe the model has already be prepared. Original error: {e}.")

    if status_message_callback:
        cb_result = status_message_callback(
            "Preparing PEFT model for training...")
        if cb_result:
            return

    lora_config_args = {
        'r': lora_r,
        'lora_alpha': lora_alpha,
        'target_modules': lora_target_modules,
        'modules_to_save': lora_modules_to_save,
        'lora_dropout': lora_dropout,
        'bias': "none",
        'task_type': "CAUSAL_LM",
    }
    config = LoraConfig(**{
        **lora_config_args,
        **(additional_lora_config or {}),
    })
    model = get_peft_model(model, config)
    if bf16:
        model = model.to(torch.bfloat16)

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
            raise ValueError(f"Checkpoint {checkpoint_name} not found")

    # Be more transparent about the % of trainable params.
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params} (calculated)"
    )
    model.print_trainable_parameters()
    if use_wandb and wandb:
        wandb.config.update({"model": {"all_params": all_params, "trainable_params": trainable_params,
                            "trainable%": 100 * trainable_params / all_params}})
    if params_info_callback:
        cb_result = params_info_callback(
            all_params=all_params, trainable_params=trainable_params)
        if cb_result:
            return

    if status_message_callback:
        cb_result = status_message_callback("Preparing train data...")
        if cb_result:
            return

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

    # If train_data is a list, convert it to datasets.Dataset
    if isinstance(train_data, list):
        with open(os.path.join(output_dir, "train_data_samples.json"), 'w') as file:
            json.dump(list(train_data[:100]), file, indent=2)
        train_data = Dataset.from_list(train_data)

    if val_set_size > 0:
        train_val = train_data.train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = train_data.shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    if status_message_callback:
        cb_result = status_message_callback("Train starting...")
        if cb_result:
            return

    # https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments
    training_args = {
        'output_dir': output_dir,
        'per_device_train_batch_size': micro_batch_size,
        'gradient_checkpointing': gradient_checkpointing,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'warmup_steps': 100,
        'num_train_epochs': num_train_epochs,
        'learning_rate': learning_rate,
        'fp16': fp16,
        'bf16': bf16,
        'logging_steps': logging_steps,
        'optim': "adamw_torch",
        'evaluation_strategy': "steps" if val_set_size > 0 else "no",
        'save_strategy': "steps",
        'eval_steps': save_steps if val_set_size > 0 else None,
        'save_steps': save_steps,
        'output_dir': output_dir,
        'save_total_limit': save_total_limit,
        'load_best_model_at_end': True if val_set_size > 0 else False,
        'ddp_find_unused_parameters': False if ddp else None,
        'group_by_length': group_by_length,
        'report_to': "wandb" if use_wandb else None,
        'run_name': wandb_run_name if use_wandb else None,
    }

    # https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        args=transformers.TrainingArguments(**{
            **training_args,
            **(additional_training_arguments or {})
        }),
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

    if use_wandb and wandb:
        wandb.finish()

    return train_output
