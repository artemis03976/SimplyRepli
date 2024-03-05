import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights


# base structure for decoder with attention in seq2seq model
class DecoderAttnBase(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, dropout):
        super(DecoderAttnBase, self).__init__()

        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.attention = BahdanauAttention(hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

        self.network = None

    def forward(self, input_seq, hidden, cell, encoder_outputs):
        embedded = self.dropout(self.embedding(input_seq))

        query = hidden[-1].unsqueeze(1)
        context, attn_weights = self.attention(query, encoder_outputs)
        embedded_attn = torch.cat((embedded, context), dim=2)

        if cell is not None:
            output, (hidden, cell) = self.network(embedded_attn, (hidden, cell))
        else:
            output, hidden = self.network(embedded_attn, hidden)
        output = self.out(output)

        return output, hidden, cell, attn_weights


# RNN backbone
class DecoderAttnRNN(DecoderAttnBase):
    def __init__(self, output_dim, embed_dim, hidden_dim, num_layers, dropout, bidirectional):
        super(DecoderAttnRNN, self).__init__(output_dim, embed_dim, hidden_dim, dropout)

        self.network = nn.RNN(embed_dim + hidden_dim, hidden_dim, num_layers,
                              batch_first=True, bidirectional=bidirectional)


# GRU backbone
class DecoderAttnGRU(DecoderAttnBase):
    def __init__(self, output_dim, embed_dim, hidden_dim, num_layers, dropout, bidirectional):
        super(DecoderAttnGRU, self).__init__(output_dim, embed_dim, hidden_dim, dropout)

        self.network = nn.GRU(embed_dim + hidden_dim, hidden_dim, num_layers,
                              batch_first=True, bidirectional=bidirectional)


# LSTM backbone
class DecoderAttnLSTM(DecoderAttnBase):
    def __init__(self, output_dim, embed_dim, hidden_dim, num_layers, dropout, bidirectional):
        super(DecoderAttnLSTM, self).__init__(output_dim, embed_dim, hidden_dim, dropout)

        self.network = nn.LSTM(embed_dim + hidden_dim, hidden_dim, num_layers,
                               batch_first=True, bidirectional=bidirectional)


if __name__ == '__main__':
    test_query = torch.randn(4, 512, 512)
    test_keys = torch.randn(4, 512, 512)

    attn = BahdanauAttention(512)

    test_context, test_weights = attn(test_query, test_keys)
    print(test_context.shape, test_weights.shape)
