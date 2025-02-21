import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram


class MelSpectrogramLoss(nn.Module):
    """
    MelSpectrogramLoss is a loss function that computes the mean absolute error between the mel spectrograms of the output and target.

    Methods:
        forward: Computes the loss.

    Args (forward):
        output (torch.Tensor): The output of the model.
        target (torch.Tensor): The target of the model.

    Returns:
        torch.Tensor: The mean absolute error between the mel spectrograms of the output and target.
    """

    def __init__(self, device: torch.device):
        super(MelSpectrogramLoss, self).__init__()
        self.device = device


    def forward(self, output: torch.Tensor, target: torch.Tensor):
        mel_spectrogram = MelSpectrogram(
            sample_rate=48000, n_fft=512, n_mels=256
        ).to(self.device)
        input_mel = mel_spectrogram(output.reshape(2, -1).to(self.device))
        target_mel = mel_spectrogram(target.reshape(2, -1).to(self.device))
        return torch.mean(torch.abs(input_mel - target_mel))


class MyMSELoss(nn.Module):
    def __init__(self):
        super(MyMSELoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, output, target):
        return self.loss(output, target)


class ESRLoss(nn.Module):
    """
    ESR loss calculates the Error-to-signal between the output/target

    ESR = 1/N * sum((target - output)^2) / (sum(target^2) + epsilon)
    """

    def __init__(self):
        super(ESRLoss, self).__init__()
        self.epsilon = 0.00001

    def forward(self, output: torch.Tensor, target: torch.Tensor):

        loss = torch.add(target, -output)
        loss = torch.pow(loss, 2)
        loss = torch.mean(loss)
        energy = torch.mean(torch.pow(target, 2)) + self.epsilon
        loss = torch.div(loss, energy)
        return loss


class DCLoss(nn.Module):
    def __init__(self):
        super(DCLoss, self).__init__()
        self.epsilon = 0.00001

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        loss = torch.pow(torch.add(torch.mean(target, 0), -torch.mean(output, 0)), 2)
        loss = torch.mean(loss)
        energy = torch.mean(torch.pow(target, 2)) + self.epsilon
        loss = torch.div(loss, energy)
        return loss


class LossWrapper(nn.Module):
    def __init__(self, losses: dict, device: torch.device):
        super(LossWrapper, self).__init__()
        self.device = device
        loss_dict = {
            "ESR": ESRLoss(),
            "DC": DCLoss(),
            "Mel": MelSpectrogramLoss(self.device),
            "MSE": nn.MSELoss(),
        }

        loss_functions = [[loss_dict[key], value] for key, value in losses.items()]

        self.loss_functions = tuple([items[0] for items in loss_functions])
        try:
            self.loss_factors = tuple(
                torch.Tensor([items[1] for items in loss_functions])
            )
        except IndexError:
            self.loss_factors = torch.ones(len(self.loss_functions))

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        loss = 0
        for i, losses in enumerate(self.loss_functions):
            loss += torch.mul(losses(output, target), self.loss_factors[i])
        return loss
