import math

import torch
import torchaudio

from deepspeaker.data.preprocess import Preprocessor


class TestPreprocessor:
    def preprocessor(self, tmp_path_factory, input, target, input_fs, target_fs):
        raw_data_dir = tmp_path_factory.mktemp(f"raw")

        torchaudio.save(
            raw_data_dir / "test-input.wav",
            input,
            input_fs,
            channels_first=False,
            encoding="PCM_F",
        )

        torchaudio.save(
            raw_data_dir / "test-target.wav",
            target,
            target_fs,
            channels_first=False,
            encoding="PCM_F",
        )

        processed_data_dir = str(tmp_path_factory.mktemp("processed"))
        p = Preprocessor(str(raw_data_dir), processed_data_dir, train_ratio=0.75)

        return p

    def test_both_48khz(self, tmp_path_factory):
        size = 1200

        p = self.preprocessor(
            tmp_path_factory,
            torch.randn([size, 2]),
            torch.randn([size, 2]),
            48000,
            48000,
        )

        assert p.resampled_fs == 48000

        train_size = int(0.75 * size)
        test_size = size - train_size

        assert p.processed_data.get("train").get("input").shape == torch.Size(
            [train_size, 2]
        )
        assert p.processed_data.get("train").get("target").shape == torch.Size(
            [train_size, 2]
        )

        assert p.processed_data.get("test").get("input").shape == torch.Size(
            [test_size, 2]
        )
        assert p.processed_data.get("test").get("target").shape == torch.Size(
            [test_size, 2]
        )

    def test_both_41khz(self, tmp_path_factory):
        size = 1200

        p = self.preprocessor(
            tmp_path_factory,
            torch.randn([size, 2]),
            torch.randn([size, 2]),
            41100,
            41100,
        )

        assert p.resampled_fs == 48000

        ratio = 48000 / 41100
        resampled_size = math.ceil(size * ratio)
        train_size = int(0.75 * resampled_size)
        test_size = resampled_size - train_size

        assert p.processed_data.get("train").get("input").shape == torch.Size(
            [train_size, 2]
        )
        assert p.processed_data.get("train").get("target").shape == torch.Size(
            [train_size, 2]
        )

        assert p.processed_data.get("test").get("input").shape == torch.Size(
            [test_size, 2]
        )
        assert p.processed_data.get("test").get("target").shape == torch.Size(
            [test_size, 2]
        )

    def test_diff_sample_rates(self, tmp_path_factory):
        size = 1200

        p = self.preprocessor(
            tmp_path_factory,
            torch.randn([1200, 2]),
            torch.randn([1200, 2]),
            48000,
            41100,
        )

        assert p.resampled_fs == 48000
        train_size = int(0.75 * size)
        test_size = size - train_size

        assert p.processed_data.get("train").get("input").shape == torch.Size(
            [train_size, 2]
        )
        assert p.processed_data.get("train").get("target").shape == torch.Size(
            [train_size, 2]
        )

        assert p.processed_data.get("test").get("input").shape == torch.Size(
            [test_size, 2]
        )
        assert p.processed_data.get("test").get("target").shape == torch.Size(
            [test_size, 2]
        )

    def test_longer_input(self, tmp_path_factory):
        input_size = 1400
        target_size = 1200

        p = self.preprocessor(
            tmp_path_factory,
            torch.randn([input_size, 2]),
            torch.randn([target_size, 2]),
            48000,
            48000,
        )

        assert p.resampled_fs == 48000

        train_size = int(0.75 * target_size)
        test_size = target_size - train_size

        assert p.processed_data.get("train").get("input").shape == torch.Size(
            [train_size, 2]
        )
        assert p.processed_data.get("train").get("target").shape == torch.Size(
            [train_size, 2]
        )

        assert p.processed_data.get("test").get("input").shape == torch.Size(
            [test_size, 2]
        )
        assert p.processed_data.get("test").get("target").shape == torch.Size(
            [test_size, 2]
        )

    def test_longer_target(self, tmp_path_factory):
        input_size = 1200
        target_size = 1400

        p = self.preprocessor(
            tmp_path_factory,
            torch.randn([input_size, 2]),
            torch.randn([target_size, 2]),
            48000,
            48000,
        )

        assert p.resampled_fs == 48000

        train_size = int(0.75 * input_size)
        test_size = input_size - train_size

        assert p.processed_data.get("train").get("input").shape == torch.Size(
            [train_size, 2]
        )
        assert p.processed_data.get("train").get("target").shape == torch.Size(
            [train_size, 2]
        )

        assert p.processed_data.get("test").get("input").shape == torch.Size(
            [test_size, 2]
        )
        assert p.processed_data.get("test").get("target").shape == torch.Size(
            [test_size, 2]
        )
