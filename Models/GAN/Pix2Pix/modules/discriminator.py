import torch
import torch.nn as nn


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, **kwargs):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, **kwargs)
        self.norm = nn.BatchNorm2d(out_channel)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.leaky_relu(x)
        return x


class PatchGANDiscriminator(nn.Module):
    def __init__(
            self,
            in_channel,
            num_layers,
            base_channel=64,
    ):
        super(PatchGANDiscriminator, self).__init__()

        out_channel = base_channel

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )

        self.discriminator = nn.ModuleList([])

        for i in range(1, num_layers):
            in_channel = out_channel
            out_channel = min(2 * out_channel, base_channel * 8)
            self.discriminator.append(
                BasicConv(in_channel, out_channel, kernel_size=4, stride=2, padding=1)
            )

        in_channel = out_channel
        out_channel = min(2 ** num_layers * base_channel, base_channel * 8)

        self.discriminator.append(
            BasicConv(in_channel, out_channel, kernel_size=4, stride=1, padding=1)
        )

        self.final_conv = nn.Conv2d(out_channel, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        x = self.in_conv(x)
        for layer in self.discriminator:
            x = layer(x)
        x = self.final_conv(x)

        return x
