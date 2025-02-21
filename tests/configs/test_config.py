from pathlib import Path
from types import SimpleNamespace

import pytest

from deepspeaker.configs.config import Config


class TestConfig:
    @pytest.fixture()
    def root_dir(self):
        return Path(__file__).parent.parent.parent / "src" / "deepspeaker"

    @pytest.fixture
    def default_config(self, root_dir):
        args = SimpleNamespace(
            **{"config_path": f"{root_dir}/configs", "config_name": "lstm_local.yaml"}
        )
        config = Config(args)
        return config

    @pytest.fixture
    def required_attributes(self):
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
            "truncated_bptt_steps",
            "num_layers",
            "hidden_size",
            "skip",
            "config_path",
            "config_name",
            "model_name",
            "model_save_path",
            "raw_data_dir",
            "processed_data_dir",
        ]

        return req_atts

    def test_default_configs(self, default_config, required_attributes):
        for att in required_attributes:
            assert hasattr(default_config, att)

    def test_missing_config_path(self, default_config, required_attributes):
        with pytest.raises(AttributeError):
            args = SimpleNamespace(**{"config_name": "lstm_local.yaml"})
            config = Config(args)

    def test_missing_config_name(self, root_dir):
        with pytest.raises(AttributeError):
            args = SimpleNamespace(**{"config_path": f"{root_dir}/configs"})
            config = Config(args)

    def test_command_line_args_override_file_args(self, root_dir):
        args = SimpleNamespace(
            **{
                "config_path": f"{root_dir}/configs",
                "config_name": "lstm_local.yaml",
                "hidden_size": 1,
            }
        )
        config = Config(args)

        assert config.hidden_size == 1

    def test_missing_argument(self, root_dir):
        with pytest.raises(AttributeError):
            args = SimpleNamespace(
                **{
                    "config_path": f"{root_dir}/configs",
                    "config_name": "fail_test.yaml",
                }
            )
            config = Config(args)
