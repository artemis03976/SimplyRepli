import torch.nn as nn


# base structure for encoder in seq2seq model
class EncoderBase(nn.Module):
    def __init__(self, input_dim, embed_dim, dropout):
        super(EncoderBase, self).__init__()

        # token embedding
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.network = None

    def forward(self, input_seq):
        embedded = self.dropout(self.embedding(input_seq))
        output, hidden = self.network(embedded)
        return output, hidden, None


# RNN backbone
class EncoderRNN(EncoderBase):
    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers, dropout, bidirectional):
        super(EncoderRNN, self).__init__(input_dim, embed_dim, dropout)

        self.network = nn.RNN(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)


# GRU backbone
class EncoderGRU(EncoderBase):
    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers, dropout, bidirectional):
        super(EncoderGRU, self).__init__(input_dim, embed_dim, dropout)

        self.network = nn.GRU(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)


# LSTM backbone
class EncoderLSTM(EncoderBase):
    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers, dropout, bidirectional):
        super(EncoderLSTM, self).__init__(input_dim, embed_dim, dropout)

        self.network = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)

    def forward(self, input_seq):
        embedded = self.dropout(self.embedding(input_seq))
        output, (hidden, cell) = self.network(embedded)
        return output, hidden, cell
