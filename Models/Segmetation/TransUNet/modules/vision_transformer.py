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

        attention_output = torch.matmul(attention_weight, value)

        attention_output = attention_output.transpose(1, 2).reshape(batch_size, -1, self.embed_dim)
        output = self.fc_out(attention_output)

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
        # self attention
        key = query = value = x
        attention_output, attention_weight = self.attention(key, query, value)
        output = self.layer_norm(x + attention_output)

        mlp_output = self.feed_forward(output)
        output = self.layer_norm(mlp_output + output)

        return output, attention_weight


class PatchEmbedding(nn.Module):
    def __init__(self, in_channel, patch_size, embed_dim, num_patches, dropout):
        super(PatchEmbedding, self).__init__()

        # use conv to patchify
        self.patchifier = nn.Sequential(
            nn.Conv2d(in_channel, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(start_dim=2),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim), requires_grad=True)

        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim), requires_grad=True)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        x = self.patchifier(x)
        x = x.permute(0, 2, 1)

        x = torch.cat([cls_token, x], dim=1)

        x = x + self.positional_embedding

        x = self.dropout(x)

        return x


class VisionTransformer(nn.Module):
    def __init__(
            self,
            in_channel,
            patch_size,
            img_size,
            num_encoders=6,
            num_heads=8,
            mlp_dim=1024,
            dropout=0.0,
    ):
        super(VisionTransformer, self).__init__()

        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = in_channel * (patch_size ** 2)

        self.patch_embedding = PatchEmbedding(in_channel, patch_size, self.embed_dim, self.num_patches, dropout)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, num_heads, mlp_dim, dropout) for _ in range(num_encoders)
        ])

    def forward(self, x):
        x = self.patch_embedding(x)

        for transformer_block in self.transformer_blocks:
            x, _ = transformer_block(x)

        return x[:, 1:, :]
