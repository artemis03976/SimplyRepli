import torch.nn as nn
import torch.nn.functional as F
from Models.RNN.models.base import RNNBaseCell, RNNBase, ClassifierBase


# Basic RNN cell for computing hidden state
class RNNCell(RNNBaseCell):
    def __init__(self, input_dim, hidden_dim, bias=True):
        super(RNNCell, self).__init__(input_dim, hidden_dim, bias=bias)

        self.input_linear = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.hidden_linear = nn.Linear(hidden_dim, hidden_dim, bias=bias)

    def forward(self, input_data, hidden_state):
        hidden, cell = hidden_state
        input_proj = self.input_linear(input_data)
        hidden_proj = self.hidden_linear(hidden)
        new_hidden = F.tanh(input_proj + hidden_proj)

        return new_hidden, cell


# RNN network
class RNN(RNNBase):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            num_layers,
            bias=True,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
    ):
        super(RNN, self).__init__(input_dim, hidden_dim, num_layers, bias, batch_first, dropout, bidirectional)

        self.cell = nn.Sequential()
        self.cell.add_module('rnnLayer1', RNNCell(input_dim, hidden_dim, bias=bias))
        for i in range(self.num_layers - 1):
            self.cell.add_module('rnnLayer%d' % (i + 2), RNNCell(hidden_dim, hidden_dim, bias=bias))

        if self.bidirectional:
            self.reversed_cell = nn.Sequential()
            self.reversed_cell.add_module('brnnLayer1', RNNCell(input_dim, hidden_dim, bias=bias))
            for i in range(self.num_layers - 1):
                self.reversed_cell.add_module('brnnLayer%d' % (i + 2), RNNCell(hidden_dim, hidden_dim, bias=bias))


class RNNClassifier(ClassifierBase):
    def __init__(
            self,
            src_vocab_size,
            embed_dim,
            hidden_dim,
            num_classes,
            num_layers=1,
            bidirectional=False,
            dropout=0.5
    ):
        super(RNNClassifier, self).__init__(
            src_vocab_size,
            embed_dim,
            hidden_dim,
            num_classes,
            num_layers,
            bidirectional,
            dropout
        )

        self.rnn = RNN(
            embed_dim,
            hidden_dim,
            num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )
