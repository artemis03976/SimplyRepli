import torch.nn as nn
import torch.nn.functional as F
from Models.RNN.models.base import RNNBaseCell, RNNBase, ClassifierBase


# Basic LSTM cell for computing hidden state
class LSTMCell(RNNBaseCell):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__(input_size, hidden_size, bias=bias)

        # W for input, U for hidden
        # input gate
        self.input_gate_w = nn.Linear(input_size, hidden_size, bias=bias)
        self.input_gate_u = nn.Linear(hidden_size, hidden_size, bias=bias)
        # output gate
        self.output_gate_w = nn.Linear(input_size, hidden_size, bias=bias)
        self.output_gate_u = nn.Linear(hidden_size, hidden_size, bias=bias)
        # forget gate
        self.forget_gate_w = nn.Linear(input_size, hidden_size, bias=bias)
        self.forget_gate_u = nn.Linear(hidden_size, hidden_size, bias=bias)
        # new info gate
        self.hidden_linear_w = nn.Linear(input_size, hidden_size, bias=bias)
        self.hidden_linear_u = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(self, input_data, hidden_state):
        hidden, cell = hidden_state
        # gate function
        input_gate = F.sigmoid(self.input_gate_w(input_data) + self.input_gate_u(hidden))
        output_gate = F.sigmoid(self.output_gate_w(input_data) + self.output_gate_u(hidden))
        forget_gate = F.sigmoid(self.forget_gate_w(input_data) + self.forget_gate_u(hidden))
        new_info = F.tanh(self.hidden_linear_w(input_data) + self.hidden_linear_u(hidden))
        # update cell and hidden
        new_cell = forget_gate * cell + input_gate * new_info
        new_hidden = output_gate * F.tanh(new_cell)

        return new_hidden, new_cell


class LSTM(RNNBase):
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
        super(LSTM, self).__init__(input_dim, hidden_dim, num_layers, bias, batch_first, dropout, bidirectional)

        # forward layer
        self.cell = nn.Sequential()
        self.cell.add_module('lstmLayer1', LSTMCell(input_dim, hidden_dim, bias=bias))
        for i in range(self.num_layers - 1):
            self.cell.add_module('lstmLayer%d' % (i + 2), LSTMCell(hidden_dim, hidden_dim, bias=bias))

        # backward layer
        if self.bidirectional:
            self.reversed_cell = nn.Sequential()
            self.reversed_cell.add_module('blstmLayer1', LSTMCell(input_dim, hidden_dim, bias=bias))
            for i in range(self.num_layers - 1):
                self.reversed_cell.add_module('blstmLayer%d' % (i + 2), LSTMCell(hidden_dim, hidden_dim, bias=bias))


class LSTMClassifier(ClassifierBase):
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
        super(LSTMClassifier, self).__init__(
            src_vocab_size,
            embed_dim,
            hidden_dim,
            num_classes,
            num_layers,
            bidirectional,
            dropout
        )

        self.rnn = LSTM(
            embed_dim,
            hidden_dim,
            num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )
