import math
import torch
import torch.nn as nn


class Normalize(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), inverse=False):
        super(Normalize, self).__init__(3, 3, kernel_size=1)

        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        sign = 1 if inverse else -1
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std

        # disable training for this module
        for p in self.parameters():
            p.requires_grad = False


class ResBlock(nn.Module):
    def __init__(self, mid_channel, with_batch_norm=False):
        super(ResBlock, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channel) if with_batch_norm else nn.Identity(),
            nn.ReLU()
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channel) if with_batch_norm else nn.Identity(),
            nn.ReLU()
        )

    def forward(self, x):
        identity = x

        x = self.conv_1(x)
        x = self.conv_2(x)

        return x + identity


class Upsample(nn.Module):
    def __init__(self, scale_factor, in_channel):
        super(Upsample, self).__init__()

        self.upsample = nn.ModuleList([])

        if (scale_factor & (scale_factor - 1)) == 0:  # scale == 2^n
            for _ in range(int(math.log(scale_factor, 2))):
                self.upsample.append(
                    nn.Sequential(
                        nn.Conv2d(in_channel, 4 * in_channel, kernel_size=3, padding=1),
                        nn.PixelShuffle(2)
                    )
                )

        elif scale_factor == 3:
            self.upsample.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, 9 * in_channel, kernel_size=3),
                    nn.PixelShuffle(3)
                )
            )
        else:
            raise NotImplementedError

    def forward(self, x):
        for layer in self.upsample:
            x = layer(x)

        return x


class EDSR(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            mid_channel,
            num_blocks,
            scale_factor,
    ):
        super(EDSR, self).__init__()

        self.normalize_layer = Normalize(1.0)
        self.denormalize_layer = Normalize(1.0, inverse=True)

        self.in_conv = nn.Conv2d(in_channel, mid_channel, kernel_size=3, padding=1)

        self.res_blocks = nn.Sequential(
            *[ResBlock(mid_channel) for _ in range(num_blocks)],
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3, padding=1)
        )

        self.out_conv = nn.Sequential(
            Upsample(scale_factor, mid_channel),
            nn.Conv2d(mid_channel, out_channel, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.normalize_layer(x)
        x = self.in_conv(x)

        x = x + self.res_blocks(x)

        x = self.out_conv(x)
        x = self.denormalize_layer(x)

        return x
