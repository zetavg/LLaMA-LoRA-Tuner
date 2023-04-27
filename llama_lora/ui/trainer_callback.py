import time
import traceback
from transformers import TrainerCallback

from ..globals import Global
from ..utils.eta_predictor import ETAPredictor


def reset_training_status():
    Global.is_train_starting = False
    Global.is_training = False
    Global.should_stop_training = False
    Global.train_started_at = time.time()
    Global.training_error_message = None
    Global.training_error_detail = None
    Global.training_total_epochs = 1
    Global.training_current_epoch = 0.0
    Global.training_total_steps = 1
    Global.training_current_step = 0
    Global.training_progress = 0.0
    Global.training_log_history = []
    Global.training_status_text = ""
    Global.training_eta_predictor = ETAPredictor()
    Global.training_eta = None
    Global.training_args = None
    Global.train_output = None
    Global.train_output_str = None
    Global.training_params_info_text = ""


def get_progress_text(current_epoch, total_epochs, last_loss):
    progress_detail = f"Epoch {current_epoch:.2f}/{total_epochs}"
    if last_loss is not None:
        progress_detail += f", Loss: {last_loss:.4f}"
    return f"Training... ({progress_detail})"


def set_train_output(output):
    end_by = 'aborted' if Global.should_stop_training else 'completed'
    result_message = f"Training {end_by}"
    Global.training_status_text = result_message

    Global.train_output = output
    Global.train_output_str = str(output)

    return result_message


def update_training_states(
        current_step, total_steps,
        current_epoch, total_epochs,
        log_history):

    Global.training_total_steps = total_steps
    Global.training_current_step = current_step
    Global.training_total_epochs = total_epochs
    Global.training_current_epoch = current_epoch
    Global.training_progress = current_step / total_steps
    Global.training_log_history = log_history
    Global.training_eta = Global.training_eta_predictor.predict_eta(current_step, total_steps)

    if Global.should_stop_training:
        return

    last_history = None
    last_loss = None
    if len(Global.training_log_history) > 0:
        last_history = log_history[-1]
        last_loss = last_history.get('loss', None)

    Global.training_status_text = get_progress_text(
        total_epochs=total_epochs,
        current_epoch=current_epoch,
        last_loss=last_loss,
    )


class UiTrainerCallback(TrainerCallback):
    def _on_progress(self, args, state, control):
        if Global.should_stop_training:
            control.should_training_stop = True

        try:
            total_steps = (
                state.max_steps if state.max_steps is not None
                else state.num_train_epochs * state.steps_per_epoch)
            current_step = state.global_step

            total_epochs = args.num_train_epochs
            current_epoch = state.epoch

            log_history = state.log_history

            update_training_states(
                total_steps=total_steps,
                current_step=current_step,
                total_epochs=total_epochs,
                current_epoch=current_epoch,
                log_history=log_history
            )
        except Exception as e:
            print("Error occurred while updating UI status:", e)
            traceback.print_exc()

    def on_epoch_begin(self, args, state, control, **kwargs):
        Global.training_args = args
        self._on_progress(args, state, control)

    def on_step_end(self, args, state, control, **kwargs):
        self._on_progress(args, state, control)
