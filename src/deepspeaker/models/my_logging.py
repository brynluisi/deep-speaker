import os

import plotly.graph_objects as go
import torch
import wandb
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def plot_compare_waveforms(
    waveform_one: torch.Tensor, waveform_two: torch.Tensor, channel: str = "Left"
):
    torch.Tensor.ndim = property(lambda self: len(self.shape))
    waveform_one = waveform_one.cpu()
    waveform_two = waveform_two.cpu()
    samples = [i for i in range(waveform_one.size(0))]

    fig = go.Figure()
    fig.update_layout(
        title=f"Target vs Output: {channel} Channel",
        xaxis_title="Samples",
        yaxis_title="Amplitude",
    )
    fig.add_trace(go.Scatter(x=samples, y=waveform_one, mode="lines", name="Target"))
    fig.add_trace(go.Scatter(x=samples, y=waveform_two, mode="lines", name="Output"))

    return fig


def create_wandb_audio_table(test_input_path, test_target_path, config, test_loss):
    # Create wandb audio table
    columns = [
        "Run Name",
        "Test Input",
        "Test Target",
        "Test Output",
        "Test Loss",
        "Loss Metric",
    ]
    data = [
        [
            wandb.run.name,
            str(test_input_path),
            str(test_target_path),
            str(os.path.join(config.model_save_path, "test_out_final.wav")),
            test_loss,
            config.loss_function,
        ],
    ]

    # create a Table with the specified columns
    table = wandb.Table(columns=columns)
    for name, ti, tt, to, loss, loss_metric in data:
        # combine song metadata and interactive media
        table.add_data(
            name,
            wandb.Audio(ti, sample_rate=48000),
            wandb.Audio(tt, sample_rate=48000),
            wandb.Audio(to, sample_rate=48000),
            loss,
            loss_metric,
        )

    return table


def plot_mel_spectrogram_comparison(target, output):

    target_left = target.squeeze(0)[:, 0].cpu().detach().numpy()
    target_right = target.squeeze(0)[:, 1].cpu().detach().numpy()
    output_left = output.squeeze(0)[:, 0].cpu().detach().numpy()
    output_right = output.squeeze(0)[:, 1].cpu().detach().numpy()

    fig, axs = plt.subplots(2, 2, figsize=(30, 10))
    target_left_spec = librosa.feature.melspectrogram(
        y=target_left, sr=48000, n_fft=2048, hop_length=1024
    )
    target_left_spec = librosa.power_to_db(target_left_spec, ref=np.max)

    target_right_spec = librosa.feature.melspectrogram(
        y=target_right, sr=48000, n_fft=2048, hop_length=1024
    )
    target_right_spec = librosa.power_to_db(target_right_spec, ref=np.max)

    output_left_spec = librosa.feature.melspectrogram(
        y=output_left, sr=48000, n_fft=2048, hop_length=1024
    )
    output_left_spec = librosa.power_to_db(output_left_spec, ref=np.max)

    output_right_spec = librosa.feature.melspectrogram(
        y=output_right, sr=48000, n_fft=2048, hop_length=1024
    )
    output_right_spec = librosa.power_to_db(output_right_spec, ref=np.max)

    im = librosa.display.specshow(
        target_left_spec, y_axis="mel", fmax=24000, x_axis="time", ax=axs[0, 0]
    )
    librosa.display.specshow(
        target_right_spec, y_axis="mel", fmax=24000, x_axis="time", ax=axs[1, 0]
    )
    librosa.display.specshow(
        output_left_spec, y_axis="mel", fmax=24000, x_axis="time", ax=axs[0, 1]
    )
    librosa.display.specshow(
        output_right_spec, y_axis="mel", fmax=24000, x_axis="time", ax=axs[1, 1]
    )

    axs[0, 0].set_title("Target Left")
    axs[1, 0].set_title("Target Right")
    axs[0, 1].set_title("Output Left")
    axs[1, 1].set_title("Output Right")

    # plt.title('Mel Spectrogram')
    fig.colorbar(axs[0, 0].collections[0], ax=axs.ravel().tolist(), format="%+2.0f dB")
    fig.suptitle("Target vs Output Mel Spectrograms")
    return fig
