import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.proj_key = nn.Linear(embed_dim, embed_dim)
        self.proj_query = nn.Linear(embed_dim, embed_dim)
        self.proj_value = nn.Linear(embed_dim, embed_dim)

        self.fc_out = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, key, query, value):
        batch_size = key.shape[0]

        # linear projection
        key = self.proj_key(key)
        query = self.proj_query(query)
        value = self.proj_value(value)

        # extract head dimension
        key = key.reshape(batch_size, -1, self.num_heads, self.head_dim)
        query = query.reshape(batch_size, -1, self.num_heads, self.head_dim)
        value = value.reshape(batch_size, -1, self.num_heads, self.head_dim)

        # transpose to get dimensions [batch_size, num_heads, seq_len, head_dim]
        key = torch.transpose(key, 1, 2)
        query = torch.transpose(query, 1, 2)
        value = torch.transpose(value, 1, 2)

        # get self attention
        energy = torch.matmul(query, key.transpose(-1, -2))
        attention_weight = torch.softmax(energy / math.sqrt(self.head_dim), dim=-1)
        attention = torch.matmul(attention_weight, value)
        # reshape to merge multi heads
        attention = attention.transpose(1, 2).reshape(batch_size, -1, self.embed_dim)
        output = self.fc_out(attention)

        return output, attention_weight


class MLPHead(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout):
        super(MLPHead, self).__init__()

        self.fc_1 = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.fc_2 = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.fc_1(x)
        x = self.fc_2(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout):
        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)

        self.feed_forward = MLPHead(embed_dim, hidden_dim, dropout)

        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # go through self attention
        key = query = value = x
        attention_output, attention_weight = self.attention(key, query, value)
        output = self.layer_norm(x + attention_output)
        # go through mlp
        mlp_output = self.feed_forward(output)
        output = self.layer_norm(mlp_output + output)

        return output, attention_weight
