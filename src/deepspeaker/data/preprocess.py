import math
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import numpy as np

from deepspeaker.data.utils import get_path


class Preprocessor:
    """
    Loads, preprocesses, and writes audio data for modeling.
    """

    def __init__(
        self,
        raw_data_dir: str,
        processed_data_dir: str,
        resampled_fs: int = 48000,
        train_ratio: float = 0.75,
    ):
        """

        Args:
            raw_data_dir: Path to directory ending in 'raw' that contains unprocessed input and target
            processed_data_dir: Path to directory ending in 'processed' that contains unprocessed input and target.
                                Directory should be on the same level as 'raw'. Directory is created if it does not exist.
            train_ratio: Ratio of raw data to keep as training data. Rest is testing data.
        """
        torchaudio.set_audio_backend("sox_io")

        self.processed_data_dir = processed_data_dir
        self.resampled_fs = resampled_fs

        input_path = get_path(raw_data_dir, "input")
        target_path = get_path(raw_data_dir, "target")
        self.file_name = input_path.stem.removesuffix("-input")

        self.processed_data = self.process(input_path, target_path, train_ratio)

    def process(self, input_path, target_path, train_ratio):
        """
        Runs processor pipeline from load to returning final tensors.

        Load >> Resample >> Trim >> Train-Test-Split
        """

        # Load input and target signals
        (
            raw_input,
            raw_input_fs,
            raw_input_channels,
        ) = self.load_wav(input_path)

        (
            raw_target,
            raw_target_fs,
            raw_target_channels,
        ) = self.load_wav(target_path)

        gain = np.floor(
            np.max(raw_input[0, :].cpu().numpy())
            / np.max(raw_target[0, :].cpu().numpy())
        )

        # Add gain
        # raw_input = self.add_gain(raw_input)
        raw_target = self.add_gain(raw_target, gain=gain)

        # Resample input and target signals
        resampled_input = self.resample_audio(raw_input, raw_input_fs)
        resampled_target = self.resample_audio(raw_target, raw_target_fs)

        # Trim signals to same length
        trimmed_input, trimmed_target = self.trim_audio(
            resampled_input, resampled_target
        )

        # Split data into training and testing
        processed_data = self.train_test_split(
            trimmed_input,
            raw_input_channels,
            trimmed_target,
            raw_target_channels,
            train_ratio,
        )

        return processed_data

    @staticmethod
    def load_wav(path) -> tuple[torch.tensor, int, int]:
        channels = torchaudio.info(path).num_channels
        waveform, fs = torchaudio.load(path, normalize=True)

        return waveform, fs, channels

    @staticmethod
    def add_gain(waveform, gain):
        """
        Adds gain to waveform.

        Args:
            waveform: Waveform to add gain to.
            gain: Gain to add to waveform.

        Returns:
            Waveform with gain added.
        """

        transform = T.Vol(gain=gain, gain_type="amplitude")
        return transform(waveform)

    def resample_audio(self, waveform, waveform_fs) -> torch.tensor:
        resampled_waveform = waveform

        if waveform_fs != self.resampled_fs:
            resampled_waveform = F.resample(waveform, waveform_fs, self.resampled_fs)

        return resampled_waveform

    @staticmethod
    def trim_audio(resampled_input, resampled_target):
        """
        Cut input or target so signals have the same length.
        """
        trimmed_input = resampled_input
        trimmed_target = resampled_target

        size_diff = resampled_target.size(1) - resampled_input.size(1)
        if size_diff > 0:
            trimmed_target = resampled_target[:, : resampled_input.size(1)]
        elif size_diff < 0:
            trimmed_input = resampled_input[:, : resampled_target.size(1)]

        return trimmed_input, trimmed_target

    @staticmethod
    def train_test_split(
        processed_input: torch.tensor,
        raw_input_channels: int,
        processed_target: torch.tensor,
        raw_target_channels: int,
        train_ratio: float,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """
        Splits input and target into two tensors each and returns a dictionary.

        Examples:
            processed_data = train_test_split(...)

            processed_data.get("train").get("input")
            processed_data.get("test").get("target")

        """
        # Handles if batch_first is True or False
        if processed_input.size(0) == raw_input_channels:
            num_samples = processed_input.size(1)
            processed_input = processed_input.permute(1, 0)
        else:
            num_samples = processed_input.size(0)
        if processed_target.size(0) == raw_target_channels:
            processed_target = processed_target.permute(1, 0)

        train_ub = math.floor(num_samples * train_ratio)

        train_input = processed_input[:train_ub, :]
        train_target = processed_target[:train_ub, :]
        test_input = processed_input[train_ub:, :]
        test_target = processed_target[train_ub:, :]

        processed_data = {
            "train": {"input": train_input, "target": train_target},
            "test": {"input": test_input, "target": test_target},
        }

        return processed_data

    def write_audio(self):
        """
        Writes processed data to the processed data directory specified during class construction.

        Data will be found in:

        <processed_data_dir>/processed/train/<file_name>-input.wav
        <processed_data_dir>/processed/train/<file_name>-target.wav

        <processed_data_dir>/processed/test/<file_name>-input.wav
        <processed_data_dir>/processed/test/<file_name>-target.wav
        """

        if not isinstance(self.processed_data_dir, Path):
            self.processed_data_dir = Path(self.processed_data_dir)

        try:
            self.processed_data_dir.mkdir()
        except FileExistsError:
            print(
                f"{self.processed_data_dir} directory already exists! Data will be overwritten."
            )

        for subset in ["train", "test"]:

            subset_dir = self.processed_data_dir / subset

            try:
                subset_dir.mkdir()
            except FileExistsError:
                print(
                    f"{subset_dir} directory already exists! Data will be overwritten"
                )
                pass

            torchaudio.save(
                subset_dir / f"{self.file_name}-{subset}-input.wav",
                self.processed_data[subset]["input"],
                self.resampled_fs,
                channels_first=False,
                encoding="PCM_F",
            )

            torchaudio.save(
                subset_dir / f"{self.file_name}-{subset}-target.wav",
                self.processed_data[subset]["target"],
                self.resampled_fs,
                channels_first=False,
                encoding="PCM_F",
            )
