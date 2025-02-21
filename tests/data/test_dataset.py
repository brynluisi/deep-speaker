import pytest
import torch
import torchaudio

from deepspeaker.data.dataset import AudioDataset


class TestAudioDataset:
    @pytest.fixture()
    def path(self, tmp_path_factory):
        path = tmp_path_factory.mktemp("data")

        input = torch.randn(1200, 2)
        target = torch.randn(1200, 2)

        torchaudio.save(
            path / "test-input.wav",
            input,
            48000,
            channels_first=False,
            encoding="PCM_F",
        )

        torchaudio.save(
            path / "test-target.wav",
            target,
            48000,
            channels_first=False,
            encoding="PCM_F",
        )

        return path

    @pytest.fixture()
    def dataset(self, path):
        dataset = AudioDataset(str(path), num_channels=2, sequence_length=100)
        return dataset

    @pytest.fixture()
    def dataset_two(self, path):
        dataset = AudioDataset(str(path), num_channels=2, sequence_length=101)
        return dataset

    def test_sequence_shapes(self, dataset):
        assert dataset.input_sequence.shape == torch.Size([12, 100, 2])
        assert dataset.target_sequence.shape == torch.Size([12, 100, 2])

    def test_sequence_shapes_two(self, dataset_two):
        assert dataset_two.input_sequence.shape == torch.Size([11, 101, 2])
        assert dataset_two.target_sequence.shape == torch.Size([11, 101, 2])
