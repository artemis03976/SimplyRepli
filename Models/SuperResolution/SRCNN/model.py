import torch
import torch.nn as nn


class SRCNN(nn.Module):
    def __init__(
            self,
            in_channel,
    ):
        super(SRCNN, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=9, padding=9 // 2),
            nn.ReLU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2),
            nn.ReLU()
        )
        self.conv_3 = nn.Conv2d(32, in_channel, kernel_size=5, padding=5 // 2)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)

        return x
