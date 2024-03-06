import torch.nn as nn
from modules.spectral_norm import SpectralNorm
from torch.nn.utils.spectral_norm import spectral_norm


class Generator(nn.Module):
    def __init__(
            self,
            latent_dim,
            feature_size,
            mid_channels,
            out_channel,
    ):
        super(Generator, self).__init__()

        self.feature_size = feature_size

        self.projection = nn.Linear(latent_dim, mid_channels[0] * feature_size * feature_size)

        self.generator = nn.ModuleList([])

        for i in range(len(mid_channels) - 1):
            self.generator.append(
                self.make_layer(mid_channels[i], mid_channels[i + 1])
            )

        self.generator.append(
            nn.Sequential(
                nn.ConvTranspose2d(mid_channels[-1], out_channel, kernel_size=3, stride=1, padding=1),
                nn.Tanh(),
            )
        )

    def make_layer(self, in_channel, out_channel):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.projection(x)
        x = x.view(x.shape[0], -1, self.feature_size, self.feature_size)

        for layer in self.generator:
            x = layer(x)

        return x


class Discriminator(nn.Module):
    def __init__(
            self,
            in_channel,
            mid_channels,
            feature_size,
    ):
        super(Discriminator, self).__init__()

        self.feature_size = feature_size

        self.discriminator = nn.ModuleList([
            self.make_layer(in_channel, mid_channels[0])
        ])

        for i in range(len(mid_channels) - 1):
            if i == len(mid_channels) - 2:
                self.discriminator.append(
                    nn.Sequential(
                        SpectralNorm(nn.Conv2d(mid_channels[-2], mid_channels[-1], kernel_size=3, stride=1, padding=1)),
                        nn.LeakyReLU(0.1),
                    )
                )
            else:
                self.discriminator.append(
                    self.make_layer(mid_channels[i], mid_channels[i + 1])
                )

        self.projection = nn.Sequential(
            nn.Flatten(),
            SpectralNorm(nn.Linear(self.feature_size * self.feature_size * mid_channels[-1], 1)),
        )

    def make_layer(self, in_channel, out_channel):
        return nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.1),
            SpectralNorm(nn.Conv2d(out_channel, out_channel, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        for layer in self.discriminator:
            x = layer(x)

        x = self.projection(x)

        return x
