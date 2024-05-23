import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    def __init__(self, channel):
        super(AttentionBlock, self).__init__()
        self.channel = channel

        self.proj_q = nn.Conv2d(channel, channel // 8, kernel_size=1)
        self.proj_k = nn.Conv2d(channel, channel // 8, kernel_size=1)
        self.proj_v = nn.Conv2d(channel, channel, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channel, height, width = x.shape
        # linear projection
        query = self.proj_q(x).view(batch_size, -1, height * width)
        key = self.proj_k(x).view(batch_size, -1, height * width)
        value = self.proj_v(x).view(batch_size, -1, height * width)
        # get self attention
        energy = torch.matmul(query.transpose(-2, -1), key)
        attention = F.softmax(energy, dim=-1)
        attention_output = torch.matmul(value, attention.transpose(-2, -1))
        attention_output = attention_output.view(batch_size, channel, height, width)
        # scale output
        attention_output = self.gamma * attention_output + x

        return attention_output
