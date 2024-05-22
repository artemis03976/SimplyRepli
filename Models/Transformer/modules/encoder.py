import torch
import torch.nn as nn

from Models.Transformer.modules.self_attention import MultiHeadAttention
from Models.Transformer.modules.feed_forward import FeedForwardNetwork


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, feed_forward_hidden_dim, dropout):
        super(TransformerEncoderBlock, self).__init__()

        self.self_attention = MultiHeadAttention(embed_dim, num_heads, dropout)

        self.feed_forward = FeedForwardNetwork(embed_dim, feed_forward_hidden_dim, dropout)

        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, input_seq, mask):
        # self attention
        key = query = value = input_seq
        self_attention_output, self_attention_weight = self.self_attention(key, query, value, mask)
        output = self.layer_norm(input_seq + self_attention_output)

        # feed forward network
        feed_forward_output = self.feed_forward(output)
        output = self.layer_norm(feed_forward_output + output)

        return output, self_attention_weight


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, feed_forward_hidden_dim, num_layers, dropout):
        super(TransformerEncoder, self).__init__()

        self.layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, feed_forward_hidden_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, input_seq, mask):
        output = input_seq
        self_attn_weight = []

        for layer in self.layers:
            output, attn_weight = layer(output, mask)
            self_attn_weight.append(attn_weight)

        return output, self_attn_weight
