import torch.nn as nn
from .blocks import ResConvBlock
from .initializer import _initializer
from .skeleton import Skeleton
from utils.snippets import *


# todo: layernorms, transformers, ffts are also considerable.
class ConvLSTM(Skeleton):
    def __init__(
        self,
        input_columns: list,
        output_columns: list,
        input_timesteps: int,
        cnn_blocks=1,
        cnn_kernel_size: int = 3,
        cnn_stride: int = 1,
        cnn_activation="gelu",
        rnn_states=200,
        rnn_layers=1,
        rnn_dropout=0,
        rnn_bidirectional=True,
        rnn_h_init: str = "learnable",
        linear_states=200,
        linear_layers=1,
        linear_activation: str = "gelu",
    ):
        super().__init__()
        self.model_info = {}  # initialize self.model_info to use update_attributes()
        self.update_attributes(locals())
        self.update_init_args(locals())

        if rnn_bidirectional:
            self.rnn_bidirectional = 2  # 1==False, 2==True
        else:
            self.rnn_bidirectional = 1
        in_channels = len(input_columns)
        n_outputs = len(output_columns)

        # CNNs
        base = 1.5
        self.CNNs = nn.ModuleList()
        for i in range(cnn_blocks):
            if i == 0:
                self.CNNs.append(
                    ResConvBlock(
                        round(in_channels * base**i),
                        round(in_channels * base ** (i + 1)),
                        round(input_timesteps / base**i),
                        round(input_timesteps / base ** (i + 1)),
                        conv_kernel_size=cnn_kernel_size,
                        conv_stride=cnn_stride,
                        activation=cnn_activation,
                        pool="max",
                    )
                )
            else:
                self.CNNs.append(
                    ResConvBlock(
                        round(in_channels * base**i),
                        round(in_channels * base ** (i + 1)),
                        round(input_timesteps / base**i),
                        round(input_timesteps / base ** (i + 1)),
                        conv_kernel_size=cnn_kernel_size,
                        conv_stride=cnn_stride,
                        activation=cnn_activation,
                        pool="avg",
                    )
                )

        # RNNs (N,L,C)->(N,Hl)
        self.RNN = nn.LSTM(
            input_size=round(in_channels * base**cnn_blocks),
            hidden_size=rnn_states,
            num_layers=rnn_layers,
            # nonlinearity='relu',# tanh, relu
            batch_first=True,
            bidirectional=rnn_bidirectional,
            dropout=rnn_dropout,
        )
        self.RNN_params = nn.ParameterDict()
        # Learnable initial hidden states
        if rnn_h_init == "learnable":
            self.RNN_params["h0"] = nn.Parameter(
                torch.zeros(
                    self.rnn_bidirectional * self.rnn_layers, 1, self.rnn_states
                )
            )
            self.RNN_params["c0"] = nn.Parameter(
                torch.zeros(
                    self.rnn_bidirectional * self.rnn_layers, 1, self.rnn_states
                )
            )
            # todo: verify initialization
            nn.init.xavier_normal_(
                self.RNN_params["h0"], gain=nn.init.calculate_gain("tanh")
            )
            nn.init.xavier_normal_(
                self.RNN_params["c0"], gain=nn.init.calculate_gain("tanh")
            )

        # Linear NNs
        activation_funcs = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "elu": nn.ELU(),
            "selu": nn.SELU(),
        }
        self.linear_activation = activation_funcs[linear_activation]
        if linear_layers == 1:
            self.linear_in = nn.Linear(self.rnn_bidirectional * rnn_states, n_outputs)
        elif linear_layers == 2:
            self.linear_in = nn.Linear(
                self.rnn_bidirectional * rnn_states, linear_states
            )
            self.linear_out = nn.Linear(linear_states, n_outputs)
        else:
            self.linear_in = nn.Linear(
                self.rnn_bidirectional * rnn_states, linear_states
            )
            self.linear_out = nn.Linear(linear_states, n_outputs)
            self.linear_hidden = nn.ModuleList()
            for i in range(linear_layers - 2):
                self.linear_hidden.append(nn.Linear(linear_states, linear_states))

        # Initialization
        self.apply(_initializer)

    def forward(self, x: torch.Tensor):
        output = x  # (N,C,L)
        #################################### CNN forward ####################################
        for rescnn in self.CNNs:
            output = rescnn.forward(output)  # (N,Cout,Lout)
        output = output.permute(0, 2, 1)  # (N,Lout,Cout)
        #################################### RNN forward ####################################
        if self.rnn_h_init == "randn":
            h0 = torch.randn(
                (self.rnn_bidirectional * self.rnn_layers, x.shape[0], self.rnn_states)
            ).to(x.device)
            c0 = torch.randn(
                (self.rnn_bidirectional * self.rnn_layers, x.shape[0], self.rnn_states)
            ).to(x.device)
        elif self.rnn_h_init == "zeros":
            h0 = torch.zeros(
                (self.rnn_bidirectional * self.rnn_layers, x.shape[0], self.rnn_states)
            ).to(x.device)
            c0 = torch.zeros(
                (self.rnn_bidirectional * self.rnn_layers, x.shape[0], self.rnn_states)
            ).to(x.device)
        elif self.rnn_h_init == "learnable":
            batch_size = x.shape[0]
            h0 = self.RNN_params["h0"].repeat(1, batch_size, 1).to(x.device)
            c0 = self.RNN_params["c0"].repeat(1, batch_size, 1).to(x.device)
        # output, hidden = self.RNN(x, (h0))  # RNN, GRU
        output, (hn, cn) = self.RNN(output, (h0, c0))  # LSTM
        output = output[:, -1, :]  # latest time step hidden state
        #################################### MLP forward ####################################
        if self.linear_layers == 1:
            output = self.linear_in(output)
        elif self.linear_layers == 2:
            output = self.linear_in(output)
            output = self.linear_activation(output)
            output = self.linear_out(output)
        elif self.linear_layers > 2:
            output = self.linear_in(output)
            output = self.linear_activation(output)
            for idx_h, h in enumerate(self.linear_hidden):
                output = h(output)
                output = self.linear_activation(output)
            output = self.linear_out(output)
        return output
