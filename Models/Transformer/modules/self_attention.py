import math
import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, head_dim):
        super(ScaledDotProductAttention, self).__init__()

        self.head_dim = head_dim

    def forward(self, key, query, value, mask):
        energy = torch.matmul(query, key.transpose(-1, -2))

        if mask is not None:
            energy = energy.masked_fill(mask, float("-1e20"))

        attention_weight = torch.softmax(energy / math.sqrt(self.head_dim), dim=-1)

        output = torch.matmul(attention_weight, value)

        return output, attention_weight


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.keys = nn.Linear(embed_dim, embed_dim)
        self.queries = nn.Linear(embed_dim, embed_dim)
        self.values = nn.Linear(embed_dim, embed_dim)

        self.self_attention = ScaledDotProductAttention(self.head_dim)

        self.fc_out = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, key, query, value, mask):
        batch_size = key.shape[0]

        # 线性投影
        k = self.keys(key)
        q = self.queries(query)
        v = self.values(value)

        # 分离多头
        k = k.reshape(batch_size, -1, self.num_heads, self.head_dim)
        q = q.reshape(batch_size, -1, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, -1, self.num_heads, self.head_dim)

        # 转置多头
        k = torch.transpose(k, 1, 2)
        q = torch.transpose(q, 1, 2)
        v = torch.transpose(v, 1, 2)

        attention_mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        # 自注意力
        attention_output, attention_weight = self.self_attention(k, q, v, attention_mask)

        attention_output = attention_output.transpose(1, 2).reshape(batch_size, -1, self.embed_dim)
        output = self.fc_out(attention_output)

        return output, attention_weight
