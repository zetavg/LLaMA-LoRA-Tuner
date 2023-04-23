import os
import json
import time
import gradio as gr
import math

from transformers import TrainerCallback
from huggingface_hub import try_to_load_from_cache, snapshot_download

from ...config import Config
from ...globals import Global
from ...models import clear_cache, unload_models
from ...utils.prompter import Prompter

from .data_processing import get_data_from_input

should_training_progress_track_tqdm = True

if Global.gpu_total_cores is not None and Global.gpu_total_cores > 2560:
    should_training_progress_track_tqdm = False


def do_train(
    # Dataset
    template,
    load_dataset_from,
    dataset_from_data_dir,
    dataset_text,
    dataset_text_format,
    dataset_plain_text_input_variables_separator,
    dataset_plain_text_input_and_output_separator,
    dataset_plain_text_data_separator,
    # Training Options
    max_seq_length,
    evaluate_data_count,
    micro_batch_size,
    gradient_accumulation_steps,
    epochs,
    learning_rate,
    train_on_inputs,
    lora_r,
    lora_alpha,
    lora_dropout,
    lora_target_modules,
    lora_modules_to_save,
    load_in_8bit,
    fp16,
    bf16,
    gradient_checkpointing,
    save_steps,
    save_total_limit,
    logging_steps,
    additional_training_arguments,
    additional_lora_config,
    model_name,
    continue_from_model,
    continue_from_checkpoint,
    progress=gr.Progress(track_tqdm=should_training_progress_track_tqdm),
):
    try:
        base_model_name = Global.base_model_name
        tokenizer_name = Global.tokenizer_name or Global.base_model_name

        resume_from_checkpoint_param = None
        if continue_from_model == "-" or continue_from_model == "None":
            continue_from_model = None
        if continue_from_checkpoint == "-" or continue_from_checkpoint == "None":
            continue_from_checkpoint = None
        if continue_from_model:
            resume_from_model_path = os.path.join(
                Config.data_dir, "lora_models", continue_from_model)
            resume_from_checkpoint_param = resume_from_model_path
            if continue_from_checkpoint:
                resume_from_checkpoint_param = os.path.join(
                    resume_from_checkpoint_param, continue_from_checkpoint)
                will_be_resume_from_checkpoint_file = os.path.join(
                    resume_from_checkpoint_param, "pytorch_model.bin")
                if not os.path.exists(will_be_resume_from_checkpoint_file):
                    raise ValueError(
                        f"Unable to resume from checkpoint {continue_from_model}/{continue_from_checkpoint}. Resuming is only possible from checkpoints stored locally in the data directory. Please ensure that the file '{will_be_resume_from_checkpoint_file}' exists.")
            else:
                will_be_resume_from_checkpoint_file = os.path.join(
                    resume_from_checkpoint_param, "adapter_model.bin")
                if not os.path.exists(will_be_resume_from_checkpoint_file):
                    # Try to get model in Hugging Face cache
                    resume_from_checkpoint_param = None
                    possible_hf_model_name = None
                    possible_model_info_file = os.path.join(
                        resume_from_model_path, "info.json")
                    if "/" in continue_from_model:
                        possible_hf_model_name = continue_from_model
                    elif os.path.exists(possible_model_info_file):
                        with open(possible_model_info_file, "r") as file:
                            model_info = json.load(file)
                            possible_hf_model_name = model_info.get(
                                "hf_model_name")
                    if possible_hf_model_name:
                        possible_hf_model_cached_path = try_to_load_from_cache(
                            possible_hf_model_name, 'adapter_model.bin')
                        if not possible_hf_model_cached_path:
                            snapshot_download(possible_hf_model_name)
                            possible_hf_model_cached_path = try_to_load_from_cache(
                                possible_hf_model_name, 'adapter_model.bin')
                        if possible_hf_model_cached_path:
                            resume_from_checkpoint_param = os.path.dirname(
                                possible_hf_model_cached_path)

                    if not resume_from_checkpoint_param:
                        raise ValueError(
                            f"Unable to continue from model {continue_from_model}. Continuation is only possible from models stored locally in the data directory. Please ensure that the file '{will_be_resume_from_checkpoint_file}' exists.")

        output_dir = os.path.join(Config.data_dir, "lora_models", model_name)
        if os.path.exists(output_dir):
            if (not os.path.isdir(output_dir)) or os.path.exists(os.path.join(output_dir, 'adapter_config.json')):
                raise ValueError(
                    f"The output directory already exists and is not empty. ({output_dir})")

        if not should_training_progress_track_tqdm:
            progress(0, desc="Preparing train data...")

        # Need RAM for training
        unload_models()
        Global.new_base_model_that_is_ready_to_be_used = None
        Global.name_of_new_base_model_that_is_ready_to_be_used = None
        clear_cache()

        prompter = Prompter(template)
        # variable_names = prompter.get_variable_names()

        data = get_data_from_input(
            load_dataset_from=load_dataset_from,
            dataset_text=dataset_text,
            dataset_text_format=dataset_text_format,
            dataset_plain_text_input_variables_separator=dataset_plain_text_input_variables_separator,
            dataset_plain_text_input_and_output_separator=dataset_plain_text_input_and_output_separator,
            dataset_plain_text_data_separator=dataset_plain_text_data_separator,
            dataset_from_data_dir=dataset_from_data_dir,
            prompter=prompter
        )

        train_data = prompter.get_train_data_from_dataset(data)

        def get_progress_text(epoch, epochs, last_loss):
            progress_detail = f"Epoch {math.ceil(epoch)}/{epochs}"
            if last_loss is not None:
                progress_detail += f", Loss: {last_loss:.4f}"
            return f"Training... ({progress_detail})"

        if Config.ui_dev_mode:
            Global.should_stop_training = False

            message = f"""Currently in UI dev mode, not doing the actual training.

Train options: {json.dumps({
    'max_seq_length': max_seq_length,
    'val_set_size': evaluate_data_count,
    'micro_batch_size': micro_batch_size,
    'gradient_accumulation_steps': gradient_accumulation_steps,
    'epochs': epochs,
    'learning_rate': learning_rate,
    'train_on_inputs': train_on_inputs,
    'lora_r': lora_r,
    'lora_alpha': lora_alpha,
    'lora_dropout': lora_dropout,
    'lora_target_modules': lora_target_modules,
    'lora_modules_to_save': lora_modules_to_save,
    'load_in_8bit': load_in_8bit,
    'fp16': fp16,
    'bf16': bf16,
    'gradient_checkpointing': gradient_checkpointing,
    'model_name': model_name,
    'continue_from_model': continue_from_model,
    'continue_from_checkpoint': continue_from_checkpoint,
    'resume_from_checkpoint_param': resume_from_checkpoint_param,
}, indent=2)}

Train data (first 10):
{json.dumps(train_data[:10], indent=2)}
            """
            print(message)

            for i in range(300):
                if (Global.should_stop_training):
                    return
                epochs = 3
                epoch = i / 100
                last_loss = None
                if (i > 20):
                    last_loss = 3 + (i - 0) * (0.5 - 3) / (300 - 0)

                progress(
                    (i, 300),
                    desc="(Simulate) " +
                    get_progress_text(epoch, epochs, last_loss)
                )

                time.sleep(0.1)

            time.sleep(2)
            return message

        if not should_training_progress_track_tqdm:
            progress(
                0, desc=f"Preparing model {base_model_name} for training...")

        log_history = []

        class UiTrainerCallback(TrainerCallback):
            def _on_progress(self, args, state, control):
                nonlocal log_history

                if Global.should_stop_training:
                    control.should_training_stop = True
                total_steps = (
                    state.max_steps if state.max_steps is not None else state.num_train_epochs * state.steps_per_epoch)
                log_history = state.log_history
                last_history = None
                last_loss = None
                if len(log_history) > 0:
                    last_history = log_history[-1]
                    last_loss = last_history.get('loss', None)

                progress_detail = f"Epoch {math.ceil(state.epoch)}/{epochs}"
                if last_loss is not None:
                    progress_detail += f", Loss: {last_loss:.4f}"

                progress(
                    (state.global_step, total_steps),
                    desc=f"Training... ({progress_detail})"
                )

            def on_epoch_begin(self, args, state, control, **kwargs):
                self._on_progress(args, state, control)

            def on_step_end(self, args, state, control, **kwargs):
                self._on_progress(args, state, control)

        training_callbacks = [UiTrainerCallback]

        Global.should_stop_training = False

        # Do not let other tqdm iterations interfere the progress reporting after training starts.
        # progress.track_tqdm = False  # setting this dynamically is not working, determining if track_tqdm should be enabled based on GPU cores at start instead.

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(os.path.join(output_dir, "info.json"), 'w') as info_json_file:
            dataset_name = "N/A (from text input)"
            if load_dataset_from == "Data Dir":
                dataset_name = dataset_from_data_dir

            info = {
                'base_model': base_model_name,
                'prompt_template': template,
                'dataset_name': dataset_name,
                'dataset_rows': len(train_data),
                'timestamp': time.time(),

                # These will be saved in another JSON file by the train function
                # 'max_seq_length': max_seq_length,
                # 'train_on_inputs': train_on_inputs,

                # 'micro_batch_size': micro_batch_size,
                # 'gradient_accumulation_steps': gradient_accumulation_steps,
                # 'epochs': epochs,
                # 'learning_rate': learning_rate,

                # 'evaluate_data_count': evaluate_data_count,

                # 'lora_r': lora_r,
                # 'lora_alpha': lora_alpha,
                # 'lora_dropout': lora_dropout,
                # 'lora_target_modules': lora_target_modules,
            }
            if continue_from_model:
                info['continued_from_model'] = continue_from_model
                if continue_from_checkpoint:
                    info['continued_from_checkpoint'] = continue_from_checkpoint

            if Global.version:
                info['tuner_version'] = Global.version

            json.dump(info, info_json_file, indent=2)

        if not should_training_progress_track_tqdm:
            progress(0, desc="Train starting...")

        wandb_group = template
        wandb_tags = [f"template:{template}"]
        if load_dataset_from == "Data Dir" and dataset_from_data_dir:
            wandb_group += f"/{dataset_from_data_dir}"
            wandb_tags.append(f"dataset:{dataset_from_data_dir}")

        train_output = Global.finetune_train_fn(
            base_model=base_model_name,
            tokenizer=tokenizer_name,
            output_dir=output_dir,
            train_data=train_data,
            # 128,  # batch_size (is not used, use gradient_accumulation_steps instead)
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            cutoff_len=max_seq_length,
            val_set_size=evaluate_data_count,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            lora_modules_to_save=lora_modules_to_save,
            train_on_inputs=train_on_inputs,
            load_in_8bit=load_in_8bit,
            fp16=fp16,
            bf16=bf16,
            gradient_checkpointing=gradient_checkpointing,
            group_by_length=False,
            resume_from_checkpoint=resume_from_checkpoint_param,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            logging_steps=logging_steps,
            additional_training_arguments=additional_training_arguments,
            additional_lora_config=additional_lora_config,
            callbacks=training_callbacks,
            wandb_api_key=Config.wandb_api_key,
            wandb_project=Config.default_wandb_project if Config.enable_wandb else None,
            wandb_group=wandb_group,
            wandb_run_name=model_name,
            wandb_tags=wandb_tags
        )

        logs_str = "\n".join([json.dumps(log)
                             for log in log_history]) or "None"

        result_message = f"Training ended:\n{str(train_output)}"
        print(result_message)
        # result_message += f"\n\nLogs:\n{logs_str}"

        clear_cache()

        return result_message

    except Exception as e:
        raise gr.Error(
            f"{e} (To dismiss this error, click the 'Abort' button)")
