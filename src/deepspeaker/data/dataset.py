import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

from deepspeaker.data.utils import get_input_target_from_path


class AudioDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        num_channels: int = 2,
        sequence_length: int = None,
        model_type: str = "RNNModel",
    ):
        """

        Args:
            data_dir: Path to directory with input and target audio files
            num_channels: Number of channels in audio files
            sequence_length: Length of sequences audio will be spliced into
            model_type: Type of model to train (rnn or cnn)
                        Data for RNNs is of shape [batch_size, sequence_length, num_channels]
                        Data for CNNs is of shape [batch_size, num_channels, sequence_length]
        """

        self.data_dir = data_dir
        self.num_channels = num_channels
        self.fs = 48000
        self.model_type = model_type

        # load audio data into tensors
        input_path, target_path = get_input_target_from_path(self.data_dir)
        self.input, self.input_fs = torchaudio.load(
            input_path, channels_first=False, normalize=True
        )
        self.target, self.target_fs = torchaudio.load(
            target_path, channels_first=False, normalize=True
        )

        # set sequence length
        if not sequence_length:
            self.sequence_length = self.target.size(0)
        else:
            self.sequence_length = sequence_length

        # split data in batches of sequences
        self.input_sequence = self.wrap_to_sequences(self.input, self.sequence_length)
        self.target_sequence = self.wrap_to_sequences(self.target, self.sequence_length)
        self._len = self.input_sequence.size(0)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return (
            self.input_sequence[index, :, :],
            self.target_sequence[index, :, :],
        )

    def wrap_to_sequences(self, data: Tensor, sequence_length: int) -> Tensor:
        """
        Args:
            data: Either input or target signal [samples, num_channels]
            sequence_length: Number of samples in training example

        Returns:
            stacked_seqs: Data packed into a sequence tensor for LSTM
                            [batch_size, sequence_length, num_channels]
        """
        num_sequences = int(np.floor(data.size(0) / sequence_length))
        self.num_sequences = num_sequences

        if self.model_type == "RNNModel":
            seqs = [
                data[(i * sequence_length) : (i * sequence_length) + sequence_length, :]
                for i in range(num_sequences)
            ]

        elif self.model_type == "WaveNetModel":
            data = data.transpose(0, 1)
            seqs = [
                data[:, (i * sequence_length) : (i * sequence_length) + sequence_length]
                for i in range(num_sequences)
            ]

        else:
            raise ValueError("Model type must be either RNNModel or WaveNetModel")

        stacked_seqs = torch.stack(seqs)
        stacked_seqs = stacked_seqs.type_as(stacked_seqs)

        return stacked_seqs


class ExpandingAudioDataset(Dataset):
    def __init__(
        self, data_dir: str, num_channels: int = 2, sequence_length: int = None
    ):
        """

        Args:
            data_dir: Path to directory with input and target audio files
            num_channels: Number of channels in audio files
            sequence_length: Length of sequences audio will be spliced into
        """

        self.data_dir = data_dir
        self.num_channels = num_channels
        self.fs = 48000

        # load audio data into tensors
        input_path, target_path = get_input_target_from_path(self.data_dir)
        self.input, self.input_fs = torchaudio.load(
            input_path, channels_first=False, normalize=True
        )
        self.target, self.target_fs = torchaudio.load(
            target_path, channels_first=False, normalize=True
        )

        # set sequence length
        if not sequence_length:
            self.sequence_length = self.target.size(0)
        else:
            self.sequence_length = sequence_length

        # split data in batches of sequences
        self.input_sequence = self.wrap_to_sequences(self.input, self.sequence_length)
        self.target_sequence = self.wrap_to_sequences(self.target, self.sequence_length)

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return (
            self.input_sequence[index],
            self.target_sequence[index],
        )

    def wrap_to_sequences(self, data: Tensor, sequence_length: int) -> list:
        """
        Args:
            data: Either input or target signal [samples, num_channels]
            sequence_length: Number of samples in training example

        Returns:
            stacked_seqs: Data packed into a sequence tensor for LSTM
                            [batch_size, sequence_length, num_channels]
        """
        num_sequences = int(np.floor(data.size(0) / sequence_length))
        self.num_sequences = num_sequences

        seqs = [
            data[0 : (i * sequence_length) + sequence_length, :]
            for i in range(num_sequences)
        ]

        return seqs
