import torch.nn as nn
import torch.nn.functional as F


# base structure for decoder in seq2seq model
class DecoderBase(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, dropout):
        super(DecoderBase, self).__init__()

        # token embedding
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

        self.network = None

    def forward(self, input_seq, hidden, cell):
        embedded = self.dropout(F.relu(self.embedding(input_seq)))
        # branch for LSTM
        if cell is not None:
            output, (hidden, cell) = self.network(embedded, (hidden, cell))
        else:
            output, hidden = self.network(embedded, hidden)
        output = self.out(output)

        return output, hidden, cell


# RNN backbone
class DecoderRNN(DecoderBase):
    def __init__(self, output_dim, embed_dim, hidden_dim, num_layers, dropout, bidirectional):
        super(DecoderRNN, self).__init__(output_dim, embed_dim, hidden_dim, dropout)

        self.network = nn.RNN(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)


# GRU backbone
class DecoderGRU(DecoderBase):
    def __init__(self, output_dim, embed_dim, hidden_dim, num_layers, dropout, bidirectional):
        super(DecoderGRU, self).__init__(output_dim, embed_dim, hidden_dim, dropout)

        self.network = nn.GRU(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)


# LSTM backbone
class DecoderLSTM(DecoderBase):
    def __init__(self, output_dim, embed_dim, hidden_dim, num_layers, dropout, bidirectional):
        super(DecoderLSTM, self).__init__(output_dim, embed_dim, hidden_dim, dropout)

        self.network = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
