import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from .initializer import _LSTMCell_initializer


class LSTMCell(nn.Module):
    """
    Self-implemented referring:
    https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html
    """

    def __repr__(self):
        return f"{self.__class__.__name__}({self.input_size}, {self.hidden_size})"

    def __init__(self, input_size, hidden_size, bias: bool = True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        # Weights input_size->hidden
        weight_ii = torch.empty(hidden_size, input_size)  # input gate
        weight_if = torch.empty(hidden_size, input_size)  # forget gate
        weight_ig = torch.empty(hidden_size, input_size)  # cell state candidate
        weight_io = torch.empty(hidden_size, input_size)  # output gate

        # Weights hidden->hidden
        weight_hi = torch.empty(hidden_size, hidden_size)  # input gate
        weight_hf = torch.empty(hidden_size, hidden_size)  # forget gate
        weight_hg = torch.empty(hidden_size, hidden_size)  # cell state candidate
        weight_ho = torch.empty(hidden_size, hidden_size)  # output gate

        self.register_parameter(
            "weight_ih",
            Parameter(torch.cat([weight_ii, weight_if, weight_ig, weight_io], dim=0)),
        )
        self.register_parameter(
            "weight_hh",
            Parameter(torch.cat([weight_hi, weight_hf, weight_hg, weight_ho], dim=0)),
        )

        # Bias
        if self.bias:
            # Bias input_size->hidden
            bias_ii = torch.empty(hidden_size)  # input gate
            bias_if = torch.empty(hidden_size)  # forget gate
            bias_ig = torch.empty(hidden_size)  # cell state candidate
            bias_io = torch.empty(hidden_size)  # output gate

            # Bias hidden->hidden
            bias_hi = torch.empty(hidden_size)  # input gate
            bias_hf = torch.empty(hidden_size)  # forget gate
            bias_hg = torch.empty(hidden_size)  # cell state candidate
            bias_ho = torch.empty(hidden_size)  # output gate

            self.register_parameter(
                "bias_ih",
                Parameter(
                    torch.cat([bias_ii, bias_if, bias_ig, bias_io], dim=0),
                    requires_grad=True,
                ),
            )
            self.register_parameter(
                "bias_hh",
                Parameter(
                    torch.cat([bias_hi, bias_hf, bias_hg, bias_ho], dim=0),
                    requires_grad=True,
                ),
            )
        self.apply(_LSTMCell_initializer)

    def forward(self, x, hx):
        # x.shape = (L=1,N,C)
        # Previous hidden, cell states
        h, c = hx
        weight_ii, weight_if, weight_ig, weight_io = torch.split(
            self.get_parameter("weight_ih"), self.hidden_size
        )
        weight_hi, weight_hf, weight_hg, weight_ho = torch.split(
            self.get_parameter("weight_hh"), self.hidden_size
        )
        bias_ii, bias_if, bias_ig, bias_io = torch.split(
            self.get_parameter("bias_ih"), self.hidden_size
        )
        bias_hi, bias_hf, bias_hg, bias_ho = torch.split(
            self.get_parameter("bias_hh"), self.hidden_size
        )

        # Input gate
        i = x @ weight_ii.T + h @ weight_hi.T
        if self.bias:
            i = i + bias_ii + bias_hi
        i = torch.sigmoid(i)

        # Forget gate
        f = x @ weight_if.T + h @ weight_hf.T
        if self.bias:
            f = f + bias_if + bias_hf
        f = torch.sigmoid(f)

        # Current cell state
        g = x @ weight_ig.T + h @ weight_hg.T
        if self.bias:
            g = g + bias_ig + bias_hg
        g = torch.tanh(g)

        # Output gate
        o = x @ weight_io.T + h @ weight_ho.T
        if self.bias:
            o = o + bias_io + bias_ho
        o = torch.sigmoid(o)

        # Next cell state
        # https://en.wikipedia.org/wiki/Hadamard_product_(matrices) == elementwise multiplication
        c_ = torch.mul(f, c) + torch.mul(
            i, g
        )  # gated previous cell state + gated current cell state
        h_ = torch.mul(o, torch.tanh(c_))  # gated next cell state
        return h_, c_
