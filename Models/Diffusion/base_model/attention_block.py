import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AttentionBlock(nn.Module):
    def __init__(self, channel, num_head=None, head_dim=None):
        super(AttentionBlock, self).__init__()

        if num_head is None:
            num_head = 1
        if head_dim is None:
            head_dim = channel // num_head

        self.num_head = num_head
        self.head_dim = head_dim

        self.proj_query = nn.Linear(channel, num_head * head_dim)
        self.proj_key = nn.Linear(channel, num_head * head_dim)
        self.proj_value = nn.Linear(channel, num_head * head_dim)

        self.final_proj = nn.Linear(num_head * head_dim, channel)

    def forward(self, x):
        batch_size, channel, height, width = x.shape

        x_reshaped = x.permute(0, 2, 3, 1).view(batch_size, -1, channel)

        query = self.proj_query(x_reshaped).view(batch_size, -1, self.num_head, self.head_dim)
        key = self.proj_key(x_reshaped).view(batch_size, -1, self.num_head, self.head_dim)
        value = self.proj_value(x_reshaped).view(batch_size, -1, self.num_head, self.head_dim)

        query = torch.transpose(query, 1, 2)
        key = torch.transpose(key, 1, 2)
        value = torch.transpose(value, 1, 2)

        energy = torch.matmul(query, key.transpose(-2, -1))
        energy = energy / (math.sqrt(channel))
        attention = F.softmax(energy, dim=-1)

        x_weighted = torch.matmul(attention, value)
        x_weighted = x_weighted.transpose(1, 2).contiguous().view(batch_size, -1, self.head_dim * self.num_head)
        x_weighted = self.final_proj(x_weighted)
        x_weighted = x_weighted.transpose(1, 2).contiguous().view(batch_size, channel, height, width)

        return x + x_weighted
