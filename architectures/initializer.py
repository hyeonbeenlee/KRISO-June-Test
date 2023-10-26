import torch.nn as nn
import numpy as np


def _initializer(layer):
    # todo: verify initializations
    if isinstance(layer, nn.Linear):  # Linear Init
        nn.init.kaiming_normal_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    elif isinstance(layer, nn.LSTM):  # LSTM Init
        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        # All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where :math:`k = \frac{1}{\text{hidden\_size}}`
        nn.init.xavier_normal_(layer.weight_ih_l0)  # tanh nonlinearity
        nn.init.xavier_normal_(layer.weight_hh_l0)  # tanh nonlinearity
        if layer.bias:
            nn.init.zeros_(layer.bias_ih_l0)
            nn.init.zeros_(layer.bias_hh_l0)
    elif isinstance(layer, nn.Conv1d):  # Conv1D Init
        nn.init.kaiming_normal_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    # elif isinstance(layer, nn.Parameter):  # Linear transformation matrix
    #     nn.init.kaiming_normal_(layer)


def _LSTMCell_initializer(module):
    for name, param in module.named_parameters():
        k = 1 / module.hidden_size
        if "weight" in name:
            nn.init.uniform_(param, -np.sqrt(k), np.sqrt(k))
        elif "bias" in name:
            nn.init.uniform_(param, -np.sqrt(k), np.sqrt(k))
