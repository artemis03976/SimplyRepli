import torch
import torch.nn as nn


class RNNBaseCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(RNNBaseCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

    def forward(self, input_data, hidden_state):
        raise NotImplementedError


class RNNBase(nn.Module):
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
        super(RNNBase, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.cell = None
        self.reversed_cell = None

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_seq):
        if self.batch_first:
            # move sequence length dim to the first position
            input_seq = input_seq.permute(1, 0, 2)

        batch_size = input_seq.size(1)
        # initialize for bidirectional RNN
        num_directions = 2 if self.bidirectional else 1

        # initialize hidden state and cell state (if LSTM) for multiple layers outputs
        hidden_state = [torch.zeros(batch_size, self.hidden_dim, device=input_seq.device)
                        for _ in range(self.num_layers * num_directions)]
        cell = [torch.zeros(batch_size, self.hidden_dim, device=input_seq.device)
                for _ in range(self.num_layers * num_directions)]

        output_seq_f = []
        output_seq_b = []

        # iterate over the sequence length
        for input_data in input_seq:
            for i, (_, layer) in enumerate(self.cell.named_children()):
                # recursive input for each layer
                hidden_state[i], cell[i] = layer(input_data, (hidden_state[i], cell[i]))
                # additionally store hidden states for output of each layer
                input_data = self.dropout(hidden_state[i]).detach()
            # store hidden states of last layer
            output_seq_f.append(hidden_state[self.num_layers - 1])

        output_seq_f = torch.stack(output_seq_f)

        if self.bidirectional:
            # reversed input sequence
            for input_data in reversed(input_seq):
                for i, (_, layer) in enumerate(self.reversed_cell.named_children()):
                    # offset for the bidirectional part
                    i += self.num_layers
                    hidden_state[i], cell[i] = layer(input_data, (hidden_state[i], cell[i]))
                    input_data = self.dropout(hidden_state[i]).detach()

                output_seq_b.append(hidden_state[self.num_layers * 2 - 1])

            output_seq_b = torch.stack(output_seq_b)
            # cat forward and backward outputs on embedding dim
            output_seq = torch.cat([output_seq_f, reversed(output_seq_b)], dim=2)
        else:
            output_seq = output_seq_f

        if self.batch_first:
            # move batch dim back
            output_seq = output_seq.permute(1, 0, 2)

        return output_seq, (hidden_state, cell)


class ClassifierBase(nn.Module):
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
        super(ClassifierBase, self).__init__()

        self.src_vocab_size = src_vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        # token embedding
        self.embedding = nn.Embedding(src_vocab_size, embed_dim)

        self.rnn = None

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, input_seq):
        # embed the input sequence
        embedded = self.embedding(input_seq)
        output_seq, hidden_state = self.rnn(embedded)
        # take the output of the last time step
        last_time_step = output_seq[:, -1]
        logits = self.classifier(last_time_step)

        return logits
