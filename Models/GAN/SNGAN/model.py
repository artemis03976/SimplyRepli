import torch.nn as nn
from Models.GAN.utilis.spectral_norm import SpectralNorm
from torch.nn.utils import spectral_norm


class Generator(nn.Module):
    def __init__(
            self,
            latent_dim,
            mid_channels,
            out_channel,
            img_size,
    ):
        super(Generator, self).__init__()

        self.feature_size = img_size // 2 ** (len(mid_channels) - 1)

        self.projection = nn.Linear(latent_dim, mid_channels[0] * self.feature_size * self.feature_size)

        self.generator = nn.ModuleList([])

        for i in range(len(mid_channels) - 1):
            if i == 0 and img_size == 28:
                self.generator.append(
                    self.make_layer(mid_channels[i], mid_channels[i + 1], kernel_size=3, stride=2, padding=0)
                )
            else:
                self.generator.append(
                    self.make_layer(mid_channels[i], mid_channels[i + 1], kernel_size=4, stride=2, padding=1)
                )

        self.generator.append(
            nn.Sequential(
                nn.Conv2d(mid_channels[-1], out_channel, kernel_size=3, stride=1, padding=1),
                nn.Tanh(),
            )
        )

    @staticmethod
    def make_layer(in_channel, out_channel, **kwargs):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, **kwargs),
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
            img_size,
    ):
        super(Discriminator, self).__init__()

        self.feature_size = img_size // 2 ** (len(mid_channels) - 1)

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

    @staticmethod
    def make_layer(in_channel, out_channel):
        # use spectral norm only in discriminator
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
