import torch.nn as nn
import torch.nn.functional as F
from Models.RNN.models.base import RNNBaseCell, RNNBase, ClassifierBase


# Basic GRU cell for computing hidden state
class GRUCell(RNNBaseCell):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__(input_size, hidden_size, bias=bias)

        # W for input, U for hidden
        # reset gate
        self.reset_gate_w = nn.Linear(input_size, hidden_size, bias=bias)
        self.reset_gate_u = nn.Linear(hidden_size, hidden_size, bias=bias)
        # update gate
        self.update_gate_w = nn.Linear(input_size, hidden_size, bias=bias)
        self.update_gate_u = nn.Linear(hidden_size, hidden_size, bias=bias)
        # new info gate
        self.hidden_linear_w = nn.Linear(input_size, hidden_size, bias=bias)
        self.hidden_linear_u = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(self, input_data, hidden_state):
        hidden, cell = hidden_state
        reset_gate = F.sigmoid(self.reset_gate_w(input_data) + self.reset_gate_u(hidden_state))
        update_gate = F.sigmoid(self.update_gate_w(input_data) + self.update_gate_u(hidden_state))
        new_info = F.tanh(self.hidden_linear_w(input_data) + reset_gate * self.hidden_linear_u(hidden_state))

        new_hidden = (1 - update_gate) * hidden_state + update_gate * new_info

        return new_hidden, cell


# GRU network
class GRU(RNNBase):
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
        super(GRU, self).__init__(input_dim, hidden_dim, num_layers, bias, batch_first, dropout, bidirectional)

        self.cell = nn.Sequential()
        self.cell.add_module('lstmLayer1', GRUCell(input_dim, hidden_dim, bias=bias))
        for i in range(self.num_layers - 1):
            self.cell.add_module('lstmLayer%d' % (i + 2), GRUCell(hidden_dim, hidden_dim, bias=bias))

        if self.bidirectional:
            self.reversed_cell = nn.Sequential()
            self.reversed_cell.add_module('blstmLayer1', GRUCell(input_dim, hidden_dim, bias=bias))
            for i in range(self.num_layers - 1):
                self.reversed_cell.add_module('blstmLayer%d' % (i + 2), GRUCell(hidden_dim, hidden_dim, bias=bias))


class GRUClassifier(ClassifierBase):
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
        super(GRUClassifier, self).__init__(
            src_vocab_size,
            embed_dim,
            hidden_dim,
            num_classes,
            num_layers,
            bidirectional,
            dropout
        )

        self.rnn = GRU(
            embed_dim,
            hidden_dim,
            num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )
