from gradio import FlaggingCallback, utils
import csv
import datetime
import os
import re
import secrets
from pathlib import Path
from typing import Any, List, Union

class CSVLogger(FlaggingCallback):
    """
    The default implementation of the FlaggingCallback abstract class. Each flagged
    sample (both the input and output data) is logged to a CSV file with headers on the machine running the gradio app.
    Example:
        import gradio as gr
        def image_classifier(inp):
            return {'cat': 0.3, 'dog': 0.7}
        demo = gr.Interface(fn=image_classifier, inputs="image", outputs="label",
                            flagging_callback=CSVLogger())
    Guides: using_flagging
    """

    def __init__(self):
        pass

    def setup(
        self,
        components: List[Any],
        flagging_dir: Union[str, Path],
    ):
        self.components = components
        self.flagging_dir = flagging_dir
        os.makedirs(flagging_dir, exist_ok=True)

    def flag(
        self,
        flag_data: List[Any],
        flag_option: str = "",
        username: Union[str, None] = None,
        filename="log.csv",
    ) -> int:
        flagging_dir = self.flagging_dir
        filename = re.sub(r"[/\\?%*:|\"<>\x7F\x00-\x1F]", "-", filename)
        log_filepath = Path(flagging_dir) / filename
        is_new = not Path(log_filepath).exists()
        headers = [
            getattr(component, "label", None) or f"component {idx}"
            for idx, component in enumerate(self.components)
        ] + [
            "flag",
            "username",
            "timestamp",
        ]

        csv_data = []
        for idx, (component, sample) in enumerate(zip(self.components, flag_data)):
            save_dir = Path(
                flagging_dir
            ) / (
                getattr(component, "label", None) or f"component {idx}"
            )
            if utils.is_update(sample):
                csv_data.append(str(sample))
            else:
                csv_data.append(
                    component.deserialize(sample, save_dir=save_dir)
                    if sample is not None
                    else ""
                )
        csv_data.append(flag_option)
        csv_data.append(username if username is not None else "")
        csv_data.append(str(datetime.datetime.now()))

        try:
            with open(log_filepath, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                if is_new:
                    writer.writerow(utils.sanitize_list_for_csv(headers))
                writer.writerow(utils.sanitize_list_for_csv(csv_data))
        except Exception as e:
            # workaround "OSError: [Errno 95] Operation not supported" with open(log_filepath, "a") on some cloud mounted directory
            random_hex = secrets.token_hex(16)
            tmp_log_filepath = str(log_filepath) + f".tmp_{random_hex}"
            with open(tmp_log_filepath, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                if is_new:
                    writer.writerow(utils.sanitize_list_for_csv(headers))
                writer.writerow(utils.sanitize_list_for_csv(csv_data))
            os.system(f"mv '{log_filepath}' '{log_filepath}.old_{random_hex}'")
            os.system(f"cat '{log_filepath}.old_{random_hex}' '{tmp_log_filepath}' > '{log_filepath}'")
            os.system(f"rm '{tmp_log_filepath}'")
            os.system(f"rm '{log_filepath}.old_{random_hex}'")

        with open(log_filepath, "r", encoding="utf-8") as csvfile:
            line_count = len([None for row in csv.reader(csvfile)]) - 1
        return line_count
