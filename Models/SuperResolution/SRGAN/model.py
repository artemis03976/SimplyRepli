import math
import torch
import torch.nn as nn


class GeneratorBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GeneratorBlock, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.PReLU()
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
        )

    def forward(self, x):
        identity = x

        x = self.conv_1(x)
        x = self.conv_2(x)

        return x + identity


class Upsample(nn.Module):
    def __init__(self, in_channel, scale_factor):
        super(Upsample, self).__init__()

        expansion = scale_factor ** 2
        self.upsample = nn.ModuleList([])

        for _ in range(int(math.log(scale_factor, 2))):
            self.upsample.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, expansion * in_channel, kernel_size=3, stride=1, padding=1),
                    nn.PixelShuffle(scale_factor),
                    nn.PReLU()
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
            num_blocks,
            scale_factor
    ):
        super(Generator, self).__init__()

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channel, base_channel, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )

        self.res_blocks = nn.ModuleList([GeneratorBlock(base_channel, base_channel) for _ in range(num_blocks)])

        self.conv_2 = nn.Sequential(
            nn.Conv2d(base_channel, base_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channel)
        )

        self.upscale = Upsample(base_channel, scale_factor)

        self.out_conv = nn.Conv2d(base_channel, out_channel, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        x = self.in_conv(x)

        identity = x

        for block in self.res_blocks:
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
