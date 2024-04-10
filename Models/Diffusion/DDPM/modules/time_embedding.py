import torch
import torch.nn as nn
import math


class TimeEmbedding(nn.Module):
    def __init__(self, time_embed_dim, max_period=10000.0):
        super(TimeEmbedding, self).__init__()

        half_dim = time_embed_dim // 2
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -(math.log(max_period) / (half_dim - 1)))

        self.register_buffer('embedding', emb.unsqueeze(0))

        self.projection = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

    def forward(self, time_steps):
        time_steps = time_steps.float()
        time_embedding = torch.matmul(time_steps.unsqueeze(1), self.embedding)
        time_embedding = torch.cat([torch.sin(time_embedding), torch.cos(time_embedding)], dim=1)
        time_embedding = self.projection(time_embedding)

        return time_embedding
