import torch
import torch.nn as nn


class RecursiveBlock(nn.Module):
    def __init__(self, in_channel, out_channel, recurse_depth=1):
        super(RecursiveBlock, self).__init__()

        self.recurse_depth = recurse_depth

        self.conv_1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)

        self.act_func = nn.ReLU()

    def forward(self, x):
        identity = x

        for i in range(self.recurse_depth):
            x = self.act_func(self.conv_1(x))
            x = self.act_func(self.conv_2(x))
            x = x + identity

        return x


class DRRN(nn.Module):
    def __init__(
            self,
            in_channel,
            mid_channel,
            num_layers,
            recurse_depth
    ):
        super(DRRN, self).__init__()

        self.in_conv = nn.Conv2d(in_channel, mid_channel, kernel_size=3, stride=1, padding=1, bias=False)

        self.recursive_blocks = nn.ModuleList([
            RecursiveBlock(mid_channel, mid_channel, recurse_depth) for _ in range(num_layers)
        ])

        self.out_conv = nn.Conv2d(mid_channel, in_channel, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        identity = x

        x = self.in_conv(x)

        for recursive_block in self.recursive_blocks:
            x = recursive_block(x)

        x = self.out_conv(x)

        x = x + identity

        return x
