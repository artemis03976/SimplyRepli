import torch
from torch import nn


class ESPCN(nn.Module):
    def __init__(self, in_channel, scale_factor):
        super(ESPCN, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(32, in_channel * (scale_factor ** 2), kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(scale_factor)
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)

        return x
