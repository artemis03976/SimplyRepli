import math
import torch
import torch.nn as nn


class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channel, growing_channel, num_dense_layers=5, res_scaling=0.2):
        super(ResidualDenseBlock, self).__init__()

        self.res_scaling = res_scaling

        self.conv_layers = nn.ModuleList([
            nn.Conv2d(in_channel + growing_channel * i, growing_channel, kernel_size=3, stride=1, padding=1)
            for i in range(num_dense_layers - 1)
        ])
        self.conv_layers.append(
            nn.Conv2d(in_channel + growing_channel * (num_dense_layers - 1), in_channel, kernel_size=3, stride=1, padding=1)
        )

        self.act_func = nn.LeakyReLU(0.2)

    def forward(self, x):
        identity = x
        dense = [x]
        for layer in self.conv_layers:
            x = torch.cat(dense, dim=1)
            x = self.act_func(layer(x))
            dense.append(x)

        return x * self.res_scaling + identity


class RRDB(nn.Module):
    def __init__(self, in_channel, growing_channel, num_dense_layers=5, num_rdb=3, res_scaling=0.2):
        super(RRDB, self).__init__()

        self.res_scaling = res_scaling

        self.rdbs = nn.ModuleList([
            ResidualDenseBlock(in_channel, growing_channel, num_dense_layers, res_scaling)
            for _ in range(num_rdb)
        ])

    def forward(self, x):
        identity = x
        for rdb in self.rdbs:
            x = rdb(x)

        return x * self.res_scaling + identity


class Upsample(nn.Module):
    def __init__(self, in_channel, scale_factor):
        super(Upsample, self).__init__()

        self.upsample = nn.ModuleList([])

        for _ in range(int(math.log(scale_factor, 2))):
            self.upsample.append(
                nn.Sequential(
                    nn.UpsamplingNearest2d(scale_factor=scale_factor),
                    nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(0.2)
                )
            )

    def forward(self, x):
        for layer in self.upsample:
            x = layer(x)

        return x


class Generator(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            base_channel,
            growing_channel,
            scale_factor,
            num_blocks,
            num_dense_layers=5,
            num_rdb=3,
            res_scaling=0.2,
    ):
        super(Generator, self).__init__()

        self.in_conv = nn.Conv2d(in_channel, base_channel, kernel_size=3, stride=1, padding=1)

        self.rrdbs = nn.ModuleList([
            RRDB(base_channel, growing_channel, num_dense_layers, num_rdb, res_scaling) for _ in range(num_blocks)
        ])

        self.conv_2 = nn.Conv2d(base_channel, base_channel, kernel_size=3, stride=1, padding=1)

        self.upscale = Upsample(base_channel, scale_factor)

        self.out_conv = nn.Sequential(
            nn.Conv2d(base_channel, base_channel, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channel, out_channel, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.in_conv(x)

        identity = x

        for block in self.rrdbs:
            x = block(x)
        x = self.conv_2(x) + identity

        x = self.upscale(x)

        x = self.out_conv(x)

        return x


class Discriminator(nn.Module):
    def __init__(
            self,
            in_channel,
            base_channel,
            num_blocks,
    ):
        super(Discriminator, self).__init__()

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channel, base_channel, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )

        self.discriminator_blocks = nn.ModuleList([
            self.make_layer(base_channel, base_channel, kernel_size=3, stride=2, padding=1),
        ])

        for _ in range(1, num_blocks // 2 + 1):
            self.discriminator_blocks.append(
                self.make_layer(base_channel, base_channel * 2, kernel_size=3, stride=1, padding=1)
            )
            self.discriminator_blocks.append(
                self.make_layer(base_channel * 2, base_channel * 2, kernel_size=3, stride=2, padding=1)
            )

            base_channel *= 2

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channel, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    @staticmethod
    def make_layer(in_channel, out_channel, **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, **kwargs),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        x = self.in_conv(x)

        for block in self.discriminator_blocks:
            x = block(x)

        x = self.classifier(x)

        return x
