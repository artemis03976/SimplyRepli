import torch
import torch.nn as nn

from Models.Transformer.modules.self_attention import MultiHeadAttention
from Models.Transformer.modules.feed_forward import FeedForwardNetwork


class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, feed_forward_hidden_dim, dropout):
        super(TransformerDecoderBlock, self).__init__()

        self.self_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(embed_dim, num_heads, dropout)

        self.feed_forward = FeedForwardNetwork(embed_dim, feed_forward_hidden_dim, dropout)

        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, input_seq, encoder_output, cross_mask, tgt_mask):
        # self attention
        identity = input_seq
        key = query = value = input_seq
        output, self_attention_weight = self.self_attention(key, query, value, tgt_mask)
        output = self.layer_norm(output + identity)

        # cross attention
        identity = output
        key = value = encoder_output
        query = output
        output, cross_attention_weight = self.cross_attention(key, query, value, cross_mask)
        output = self.layer_norm(output + identity)

        # feed forward
        identity = output
        output = self.feed_forward(output)
        output = self.layer_norm(output + identity)

        return output, self_attention_weight, cross_attention_weight


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, feed_forward_hidden_dim, num_layers, dropout):
        super(TransformerDecoder, self).__init__()

        self.layers = nn.ModuleList([
            TransformerDecoderBlock(embed_dim, num_heads, feed_forward_hidden_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, input_seq, encoder_output, cross_mask, tgt_mask):
        output = input_seq
        self_attn_weights = []
        cross_attn_weights = []

        for layer in self.layers:
            output, self_attn_weight, cross_attn_weight = layer(output, encoder_output, cross_mask, tgt_mask)
            self_attn_weights.append(self_attn_weight)
            cross_attn_weights.append(cross_attn_weight)

        return output, self_attn_weights, cross_attn_weights
