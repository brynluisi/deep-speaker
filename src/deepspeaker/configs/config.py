import os
from argparse import Namespace
from pathlib import Path

import yaml
import uuid


class Config:
    def __init__(self, args: Namespace):

        file_config = self.load_config(args.config_path, args.config_name)
        full_config = self.merge_file_and_args_configs(file_config, args)

        for key in full_config:
            setattr(self, key, full_config[key])

        id = uuid.uuid4()
        self.model_name = f"{self.data_name}-{self.model}-{id}"
        self.model_save_path = f"{self.save_location}/{self.model_name}"
        self.raw_data_dir = f"{self.data_dir}/raw/{self.data_name}"
        self.processed_data_dir = f"{self.data_dir}/processed/{self.data_name}"

    def __post_init__(self):
        req_atts = [
            "data_dir",
            "data_name",
            "save_location",
            "project_name",
            "input_channels",
            "output_channels",
            "sequence_length",
            "epochs",
            "device",
            "learn_rate",
            "loss_function",
            "model",
            "batch_size",
            "config_path",
            "config_name",
            "model_name",
            "model_save_path",
            "raw_data_dir",
            "processed_data_dir",
        ]
        for att in req_atts:
            getattr(self, att)

    # Function to load yaml configuration file
    @staticmethod
    def load_config(config_path: str, config_name: str) -> dict:
        with open(os.path.join(config_path, config_name)) as file:
            config = yaml.safe_load(file)

        return config

    @staticmethod
    def merge_file_and_args_configs(file_config: dict, args: Namespace) -> dict:
        # read file

        argument_config = vars(args)
        full_config = file_config

        for key, val in argument_config.items():
            if val is not None:
                full_config.update({key: val})

        # full_config contains all arguments from the config file and arguments from the command line
        # Command line arguments override file configuration even if the config file contains a
        # different value
        return full_config

    def save_config(self) -> None:
        # Save model config to save path
        p = Path(self.model_save_path)
        p.mkdir(parents=True, exist_ok=True)
        with open(f"{self.model_save_path}/config.yaml", "w", encoding="utf8") as f:
            # json.dump(vars(self), f, ensure_ascii=True)
            yaml.dump(vars(self), f)
