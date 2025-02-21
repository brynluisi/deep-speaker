import itertools
import os
import time
from typing import Optional, Union, Tuple, Any

import numpy as np
import pytorch_lightning as pl
import torch
import torchaudio
import wandb
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch import optim, nn, Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split

from deepspeaker.data.dataset import ExpandingAudioDataset, AudioDataset
from deepspeaker.models.loss_functions import ESRLoss, MelSpectrogramLoss, DCLoss, MyMSELoss, LossWrapper
from deepspeaker.models.my_logging import (
    plot_compare_waveforms,
    plot_mel_spectrogram_comparison,
)


class AudioDataModule(pl.LightningDataModule):
    """
    Defines train, val, and test dataloaders.
    """

    def __init__(
        self,
        data_dir: str = "./",
        sequence_length: int = 24000,
        batch_size: int = 50,
        model_type: str = "RNNModel",
        num_channels: int = 2,
        expanding: bool = False,
    ):
        super().__init__()
        self.test_split = None
        self.train_split = None
        self.val_split = None
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.model_type = model_type
        self.num_channels = num_channels
        self.expanding = expanding

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            if self.expanding:
                train_data = ExpandingAudioDataset(
                    self.data_dir, self.num_channels, self.sequence_length
                )
            else:
                train_data = AudioDataset(
                    self.data_dir,
                    self.num_channels,
                    self.sequence_length,
                    self.model_type,
                )

            # Split training dataset into train and val sets
            train_size = int(len(train_data) * 0.75)
            val_size = len(train_data) - train_size
            self.train_split, self.val_split = random_split(
                train_data, [train_size, val_size], generator=torch.Generator().manual_seed(42)
            )

        # Assign test dataset for use in dataloaders
        if stage == "test" or stage is None:
            if self.expanding:
                self.test_split = ExpandingAudioDataset(
                    self.data_dir, self.num_channels, self.sequence_length
                )
            else:
                self.test_split = AudioDataset(
                    self.data_dir,
                    self.num_channels,
                    self.sequence_length,
                    self.model_type,
                )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_loader = DataLoader(
            self.train_split,
            batch_size=1 if self.expanding else self.batch_size,
            num_workers=1,
            pin_memory=True,
        )
        return train_loader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        val_loader = DataLoader(
            self.val_split,
            batch_size=1 if self.expanding else self.batch_size,
            num_workers=1,
            pin_memory=True,
        )
        return val_loader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        test_loader = DataLoader(
            self.test_split,
            batch_size=1 if self.expanding else self.batch_size,
            num_workers=1,
            pin_memory=True,
        )
        return test_loader


"""
RNN Model 

This model is based on the paper: https://www.mdpi.com/2076-3417/10/3/766
refactored from: https://github.com/Alec-Wright/Automated-GuitarAmpModelling
"""


class RNNModel(pl.LightningModule):
    """
    Defines lstm model for speaker emulation.

    Model consists of a single LSTM block (with a variable number of layers)
    followed by a fully-connected layer.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        batch_size: int,
        hidden_size: int,
        sequence_length: int,
        cell_type: str,
        num_layers: int,
        bidirectional: bool,
        criterion: nn.Module,
        learning_rate: int,
        save_path: str,
        truncated_bptt_steps: Optional[int],
        skip: bool = False,
    ):
        """

        Args:
            input_size: Number of input channels
            output_size: Number of output channels
            batch_size: Number of Sequences in a batch
            hidden_size: Size of the hidden vector in LSTM block
            sequence_length: Number of samples in a sequence
            cell_type: Type of RNN cell to use (lstm or gru)
            num_layers: Number of layers in the LSTM block
            truncated_bptt_steps: Perform backpropagation every k steps of a
            much longer sequence (https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html?highlight=truncated_bptt_steps#truncated-bptt-steps)
            criterion: The loss function
            learning_rate:
            save_path: Location of audio, figures, to be saved during training
            skip: Include a skip connection to output of fully-connected layer or not
        """
        super(RNNModel, self).__init__()

        # define hyperparameters
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.skip = skip
        self.learning_rate = learning_rate
        self.save_hyperparameters(ignore=["criterion"])

        self.loss_functions = {
            "MSE": MyMSELoss(),
            "ESR": ESRLoss(),
            "Mel": MelSpectrogramLoss(device=self.device),
            "DC": DCLoss()
        }

        self.truncated_bptt_steps = truncated_bptt_steps

        # Define other model specific attributes
        self.save_path = save_path
        self.criterion = criterion
        # for onnx export: shows In sizes and Out sizes during sanity check
        self.example_input_array = torch.randn(
            self.batch_size, self.sequence_length, self.input_size
        )
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.d = 2
        else:
            self.d = 1

        # Define layers of the model
        if self.cell_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                bidirectional=self.bidirectional,
            )
        elif self.cell_type == "gru":
            self.rnn = nn.GRU(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                bidirectional=self.bidirectional,
            )
        else:
            raise ValueError("Invalid cell type")

        self.fc = nn.Linear(self.d * self.hidden_size, self.output_size)

    def init_hidden(self, batch_size: int) -> Union[tuple[Tensor, Tensor], Tensor]:
        """

        Args:
             batch_size: Number of sequences in a batch
        Returns:
            hiddens: Tuple of hidden state and cell state Tensors: (h0, c0)
                        or
                     Tensor of hidden state: h0

        """

        if self.cell_type == "lstm":
            hiddens = (
                torch.zeros(self.num_layers * self.d, batch_size, self.hidden_size),
                torch.zeros(self.num_layers * self.d, batch_size, self.hidden_size),
            )

        elif self.cell_type == "gru":
            hiddens = torch.zeros(
                self.num_layers * self.d, batch_size, self.hidden_size
            )

        else:
            raise ValueError("Cell type must be either lstm or gru")

        return hiddens

    def forward(
        self, input: Tensor, hiddens=None
    ) -> tuple[Tensor, Union[tuple[Tensor, Tensor], Tensor]]:
        """

        Args:
            input: (batch_size, sequence_length, input_size)
            hiddens: (h0, c0) or h0

                h0: (D * num_layers, batch_size, hidden_size)
                c0: (D * num_layers, batch_size, hidden_size)

        Returns:
            output: (batch_size, sequence_length, input_size)
            hiddens: (h0, c0) or h0

                h0: (D * num_layers, batch_size, hidden_size)
                c0: (D * num_layers, batch_size, hidden_size)

        """

        if hiddens is None:
            hiddens = self.init_hidden(input.size(0))

        if self.cell_type == "lstm":
            h0 = hiddens[0].type_as(input)
            c0 = hiddens[1].type_as(input)

            output, hidden = self.rnn(input, (h0.detach(), c0.detach()))

        elif self.cell_type == "gru":
            h0 = hiddens.type_as(input)
            output, hiddens = self.rnn(input, h0.detach())

        else:
            raise ValueError("Cell type must be either lstm or gru")

        output = self.fc(output)

        if self.skip:
            identity = input
            output += identity

        if self.truncated_bptt_steps:
            return output, hiddens
        else:
            return output

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer=optimizer, patience=1, factor=0.5),
                "monitor": "val_loss",
                "frequency": 1,
            },
        }

    def training_step(
        self,
        batch: tuple[Tensor, Tensor],
        batch_idx: int,
        hiddens: tuple[Tensor, Tensor] = None,
    ) -> dict[str, Union[Tensor, tuple[Tensor, Tensor]]]:

        input, target = batch

        if self.truncated_bptt_steps:
            output, hiddens = self(input, hiddens)
            loss = self.criterion(output=output, target=target)
            # Returning hiddens is required for truncated_bptt_steps
            loss_dict = {"loss": loss, "hiddens": hiddens}


        else:
            output = self(input)
            loss = self.criterion(output=output, target=target)
            loss_dict = {"loss": loss}

        for loss_name, loss_function in self.loss_functions.items():
            left_loss = loss_function(output=output[:, :, 0], target=target[:, :, 0])
            right_loss = loss_function(output=output[:, :, 1], target=target[:, :, 1])
            loss_dict[f"{loss_name}_loss_left"] = left_loss
            loss_dict[f"{loss_name}_loss_right"] = right_loss
        return loss_dict

    def validation_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> dict[str, Tensor]:

        input, target = batch
        if self.truncated_bptt_steps:
            output, hiddens = self(input)
        else:
            output = self(input)

        loss = self.criterion(output=output, target=target)
        loss_dict = {"loss": loss}
        for loss_name, loss_function in self.loss_functions.items():
            left_loss = loss_function(output=output[:, :, 0], target=target[:, :, 0])
            right_loss = loss_function(output=output[:, :, 1], target=target[:, :, 1])
            loss_dict[f"{loss_name}_loss_left"] = left_loss
            loss_dict[f"{loss_name}_loss_right"] = right_loss
        del target, output
        return loss_dict

    def training_epoch_end(self, train_step_outputs: list[dict]) -> None:
        if self.truncated_bptt_steps:
            train_step_outputs = list(itertools.chain.from_iterable(train_step_outputs))
        avg_train_loss = torch.tensor([x["loss"] for x in train_step_outputs]).mean()
        self.log("train_loss", avg_train_loss)
        for loss_name, loss_function in self.loss_functions.items():
            avg_train_loss_left = torch.tensor([x[f"{loss_name}_loss_left"] for x in train_step_outputs]).mean()
            avg_train_loss_right = torch.tensor([x[f"{loss_name}_loss_right"] for x in train_step_outputs]).mean()
            self.log(f"train_{loss_name}_loss_left", avg_train_loss_left)
            self.log(f"train_{loss_name}_loss_right", avg_train_loss_right)

    def validation_epoch_end(self, val_step_outputs: list[dict]) -> dict[str, Tensor]:
        avg_val_loss = torch.tensor([x["loss"] for x in val_step_outputs]).mean()
        self.log("val_loss", avg_val_loss)
        for loss_name, loss_function in self.loss_functions.items():
            avg_val_loss_left = torch.tensor([x[f"{loss_name}_loss_left"] for x in val_step_outputs]).mean()
            avg_val_loss_right = torch.tensor([x[f"{loss_name}_loss_right"] for x in val_step_outputs]).mean()
            self.log(f"val_{loss_name}_loss_left", avg_val_loss_left)
            self.log(f"val_{loss_name}_loss_right", avg_val_loss_right)
        return {"val_loss": avg_val_loss}

    def test_step(
        self,
        batch: tuple[Tensor, Tensor],
        batch_idx: int,
    ) -> tuple[Tensor, Tensor]:

        input, target = batch
        if self.truncated_bptt_steps:
            output, hiddens = self(input)
        else:
            output = self(input)

        return target, output

    def test_epoch_end(self, test_step_outputs: list[dict]) -> dict[str, Tensor]:
        target, output = test_step_outputs[0]
        loss = self.criterion(output=output, target=target)
        self.log("test_loss", loss)

        for loss_name, loss_function in self.loss_functions.items():
            left_loss = loss_function(output=output[:, :, 0], target=target[:, :, 0])
            right_loss = loss_function(output=output[:, :, 1], target=target[:, :, 1])
            self.log(f"test_left_{loss_name}", left_loss, on_epoch=True, logger=True)
            self.log(f"test_right_{loss_name}", right_loss, on_epoch=True, logger=True)
        # Plot Mel Spectrogram
        mel_spec_fig = plot_mel_spectrogram_comparison(target, output)
        wandb.log({"mel_specs": [wandb.Image(mel_spec_fig)]})

        # Plot small set of samples in time-domain for model evaluation
        flat_target = target.squeeze(0)
        flat_output = output.squeeze(0)

        fig = plot_compare_waveforms(
            flat_target[100000:105000, 0], flat_output[100000:105000, 0], "Left"
        )
        wandb.log({"Left Channel Waveform Chart": fig})

        fig = plot_compare_waveforms(
            flat_target[100000:105000, 1], flat_output[100000:105000, 1], "Right"
        )
        wandb.log({"Right Channel Waveform Chart": fig})

        # Save output of model on the test set using the model produced by the end of
        # the training loop
        torchaudio.save(
            os.path.join(self.save_path, "test_out_final.wav"),
            flat_output.cpu(),
            48000,
            encoding="PCM_F",
            format="wav",
            channels_first=False,
        )
        
        return {"loss": loss}


"""
WaveNet-style Model 

based on this paper: https://www.mdpi.com/2076-3417/10/3/766/htm
refactored version of: https://github.com/GuitarML/pedalnet

"""


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,  # controls the spacing between the kernel points
        groups: int = 1,  # all inputs are convolved to all outputs
        bias: bool = True,
    ):

        super(CausalConv1d, self).__init__()
        self.__padding = (
            kernel_size - 1
        ) * dilation  # one sided padding so convolution is causal
        self.in_channels = in_channels

        self.conv1d = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input: Tensor) -> Tensor:
        """

        Args:
            input: (batch_size, in_channels, sequence_length)

        Returns:
            output: (batch_size, out_channels, sequence_length)

            in general Lout = [[sequence_length + 2 * padding - dilation
                    * (kernel_size - 1) - 1] / stride] + 1
            but padding is removed here so Lout = sequence_length

        """
        result = self.conv1d(input)
        if self.__padding != 0:
            return result[:, :, : -self.__padding]
        return result


def _conv_stack(dilations, in_channels, out_channels, kernel_size):
    """
    Stack of dilated convolutional layers
    """
    return nn.ModuleList(
        [
            CausalConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                dilation=d,
                kernel_size=kernel_size,
            )
            for i, d in enumerate(dilations)
        ]
    )


class WaveNet(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        num_channels,
        dilation_depth,
        num_blocks,
        kernel_size,
    ):
        """
        WaveNet-style model

        Args:
            input_channels: Number of input channels for audio (1 for mono, 2 for stereo)
            output_channels: Number of output channels for audio (1 for mono, 2 for stereo)
            num_channels: Number of output channels for the input layer and input channels for a dilated causal convolution block
            dilation_depth: number of "hidden" layers in a dilated causal convolution block
            num_blocks: Number dilated causal convolution blocks
            kernel_size: kernel size for the dilated convolutions
        """
        super(WaveNet, self).__init__()
        self.num_channels = num_channels
        dilations = [2**d for d in range(dilation_depth)] * num_blocks
        internal_channels = int(num_channels * 2)  #
        self.hidden = _conv_stack(
            dilations, num_channels, internal_channels, kernel_size
        )
        self.residuals = _conv_stack(dilations, num_channels, num_channels, 1)
        self.input_layer = CausalConv1d(
            in_channels=input_channels,  # 1 for mono 2 for stereo
            out_channels=num_channels,
            kernel_size=1,
        )

        self.linear_mix = nn.Conv1d(
            in_channels=num_channels * dilation_depth * num_blocks,
            out_channels=output_channels,  # 1 for mono 2 for stereo
            kernel_size=1,
        )

    def forward(self, x):
        out = x
        skips = []
        out = self.input_layer(out)

        for hidden, residual in zip(self.hidden, self.residuals):
            x = out
            out_hidden = hidden(x)

            # gated activation
            #   split (50,16,3) into two (50,8,3) for tanh and sigmoid calculations
            out_hidden_split = torch.split(out_hidden, self.num_channels, dim=1)
            out = torch.tanh(out_hidden_split[0]) * torch.sigmoid(out_hidden_split[1])

            skips.append(
                out
            )  # save skip connection for every block * layer [num_blocks * dilation_depth]

            out = residual(out)
            out = out + x

        out = torch.cat(skips, dim=1)
        out = self.linear_mix(out)
        return out


class WaveNetModel(pl.LightningModule):
    """
    Lightning Module for Wavenet Model
    """

    def __init__(
        self,
        criterion,
        learning_rate,
        save_path,
        input_channels,
        output_channels,
        num_channels,
        dilation_depth,
        num_blocks,
        kernel_size,
    ):

        super(WaveNetModel, self).__init__()
        self.criterion = criterion
        self.save_path = save_path
        self.learning_rate = learning_rate
        self.num_channels = num_channels
        self.dilation_depth = dilation_depth
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.wavenet = WaveNet(
            input_channels=input_channels,
            output_channels=output_channels,
            num_channels=self.num_channels,
            dilation_depth=self.dilation_depth,
            num_blocks=self.num_blocks,
            kernel_size=self.kernel_size,
        )
        self.example_input_array = torch.zeros(1, input_channels, 24000)
        self.save_hyperparameters(ignore=["criterion"])

        self.loss_functions = {
            "MSE": MyMSELoss(),
            "ESR": ESRLoss(),
            "Mel": MelSpectrogramLoss(device=self.device),
            "DC": DCLoss()
        }

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer=optimizer, patience=1, factor=0.5),
                "monitor": "val_loss",
                "frequency": 1,
            },
        }

    def forward(self, x):
        return self.wavenet(x)

    def training_step(self, batch, batch_idx):
        input, target = batch
        output = self.forward(input)
        loss = self.criterion(output=output, target=target)

        loss_dict = {"loss": loss}

        for loss_name, loss_function in self.loss_functions.items():
            left_loss = loss_function(output=output[:, 0, :], target=target[:, 0, :]).item()
            right_loss = loss_function(output=output[:, 1, :], target=target[:, 1, :]).item()
            loss_dict[f"{loss_name}_loss_left"] = left_loss
            loss_dict[f"{loss_name}_loss_right"] = right_loss

        return loss_dict

    def training_epoch_end(self, train_step_outputs: list[dict]) -> None:
        avg_train_loss = torch.tensor([x["loss"].item() for x in train_step_outputs]).mean()
        self.log("train_loss", avg_train_loss)
        for loss_name, loss_function in self.loss_functions.items():
            avg_train_loss_left = torch.tensor([x[f"{loss_name}_loss_left"] for x in train_step_outputs]).mean()
            avg_train_loss_right = torch.tensor([x[f"{loss_name}_loss_right"] for x in train_step_outputs]).mean()
            self.log(f"train_{loss_name}_loss_left", avg_train_loss_left)
            self.log(f"train_{loss_name}_loss_right", avg_train_loss_right)


    def validation_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> dict[str, Tensor]:

        input, target = batch
        output = self.forward(input)

        loss = self.criterion(output=output, target=target)
        loss_dict = {"loss": loss}

        for loss_name, loss_function in self.loss_functions.items():
            left_loss = loss_function(output=output[:, 0, :], target=target[:, 0, :])
            right_loss = loss_function(output=output[:, 1, :], target=target[:, 1, :])
            loss_dict[f"{loss_name}_loss_left"] = left_loss
            loss_dict[f"{loss_name}_loss_right"] = right_loss

        return loss_dict

    def validation_epoch_end(self, val_step_outputs: list[dict]) -> dict[str, Tensor]:
        avg_val_loss = torch.tensor([x["loss"].item() for x in val_step_outputs]).mean()
        self.log("val_loss", avg_val_loss)

        for loss_name, loss_function in self.loss_functions.items():
            avg_val_loss_left = torch.tensor([x[f"{loss_name}_loss_left"] for x in val_step_outputs]).mean()
            avg_val_loss_right = torch.tensor([x[f"{loss_name}_loss_right"] for x in val_step_outputs]).mean()
            self.log(f"val_{loss_name}_loss_left", avg_val_loss_left)
            self.log(f"val_{loss_name}_loss_right", avg_val_loss_right)

        return {"val_loss": avg_val_loss}

    def test_step(
        self,
        batch: tuple[Tensor, Tensor],
        batch_idx: int,
    ) -> tuple[Tensor, Any]:

        input, target = batch
        output = self.forward(input)
        return target, output

    def test_epoch_end(self, test_step_outputs: list[dict]) -> dict[str, Tensor]:
        # do something with the outputs of all test batches
        target, output = test_step_outputs[0]
        loss = self.criterion(output=output, target=target)
        self.log("test_loss", loss.item())

        for loss_name, loss_function in self.loss_functions.items():
            left_loss = loss_function(output=output[:, 0, :], target=target[:, 0, :]).item()
            right_loss = loss_function(output=output[:, 1, :], target=target[:, 1, :]).item()

            self.log(f"test_left_{loss_name}", left_loss, on_epoch=True, logger=True)
            self.log(f"test_right_{loss_name}", right_loss, on_epoch=True, logger=True)

        output = output.transpose(1, 2)
        target = target.transpose(1, 2)

        mel_spec_fig = plot_mel_spectrogram_comparison(target, output)

        wandb.log({"mel_specs": [wandb.Image(mel_spec_fig)]})


        # Plot small set of samples for model evaluation
        flat_target = target.squeeze(0)
        flat_output = output.squeeze(0)

        left_fig = plot_compare_waveforms(
            flat_target[100000:105000, 0], flat_output[100000:105000, 0], "Left"
        )
        wandb.log({"Left Channel Waveform Chart": left_fig})

        right_fig = plot_compare_waveforms(
            flat_target[100000:105000, 1], flat_output[100000:105000, 1], "Right"
        )
        wandb.log({"Right Channel Waveform Chart": right_fig})

        # Save output of model on the test set using the model produced by the end of
        # the training loop
        torchaudio.save(
            os.path.join(self.save_path, "test_out_final.wav"),
            flat_output.cpu(),
            48000,
            encoding="PCM_F",
            format="wav",
            channels_first=False,
        )

        return {"loss": loss}
