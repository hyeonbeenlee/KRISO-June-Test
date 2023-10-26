import torch.nn as nn
import warnings
from utils.snippets import *


class ResConvBlock(nn.Module):
    """
    https://pseudo-lab.github.io/pytorch-guide/docs/ch03-1.html
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        input_timesteps,
        output_timesteps,
        conv_kernel_size: int = 3,
        conv_stride: int = 1,
        activation: str = "gelu",
        pool: str = "max",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        self.out_channels = out_channels

        activation_funcs = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "elu": nn.ELU(),
            "selu": nn.SELU(),
        }
        pool_layers = {
            "max": nn.AdaptiveMaxPool1d(self.output_timesteps),
            "avg": nn.AdaptiveAvgPool1d(self.output_timesteps),
        }
        self.activation = activation_funcs[activation]
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=conv_kernel_size,
            stride=conv_stride,
            bias=True,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=1, stride=1, bias=True
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.pool = pool_layers[pool]

        with torch.no_grad():
            test_input = torch.rand(1, self.in_channels, self.input_timesteps)
            simple_forward = lambda x: self.conv2(self.conv1(x))
            test_output = simple_forward(test_input)
        self.Lout = test_output.shape[-1]
        self.verify_timesteps()

        self.w1 = nn.Parameter(torch.zeros(1, in_channels, out_channels))
        self.w2 = nn.Parameter(torch.zeros(1, input_timesteps, self.Lout))
        self.bn3 = nn.BatchNorm1d(out_channels)
        nn.init.kaiming_normal_(self.w1)
        nn.init.kaiming_normal_(self.w2)

    def verify_timesteps(self):
        if self.Lout < self.output_timesteps:
            warnings.warn(
                f"Timesteps are being over-pooled: {self.Lout}->{self.output_timesteps}",
                UserWarning,
            )

    def downsample(self, x):
        out = torch.einsum("NCL,xCc,yLl->Ncl", x, self.w1, self.w2)
        out = self.bn3(out)
        return out

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        residual = self.downsample(x)
        out += residual
        out = self.activation(out)

        out = self.pool(out)
        return out
