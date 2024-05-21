import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_channel, latent_dim, mid_channels, img_size, kernel_size=3):
        super(Encoder, self).__init__()

        self.img_size = img_size

        encoder = [
            nn.Conv2d(in_channel, mid_channels[0], kernel_size=kernel_size),
            nn.ReLU(),
        ]

        self.img_size -= kernel_size // 2

        for i in range(len(mid_channels) - 1):
            encoder += [
                nn.Conv2d(mid_channels[i], mid_channels[i + 1], kernel_size=kernel_size),
                nn.BatchNorm2d(mid_channels[i + 1]),
                nn.ReLU(),
            ]
            self.img_size -= kernel_size // 2

        # projection into linear latent dim
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(mid_channels[-1], latent_dim * 2),
        )

        self.encoder = nn.Sequential(*encoder)

    def forward(self, x):
        x = self.encoder(x)
        x = self.projection(x)

        return x


class Decoder(nn.Module):
    def __init__(self, out_channel, latent_dim, mid_channels, feature_size, kernel_size=3):
        super(Decoder, self).__init__()

        self.feature_size = feature_size
        self.mid_channels = mid_channels

        # projection back into conv space
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, mid_channels[-1]),
            nn.Linear(mid_channels[-1], mid_channels[-1] * feature_size * feature_size),
        )

        decoder = []

        for i in reversed(range(len(mid_channels) - 1)):
            decoder += [
                nn.ConvTranspose2d(mid_channels[i + 1], mid_channels[i], kernel_size=kernel_size),
                nn.BatchNorm2d(mid_channels[i]),
                nn.ReLU(),
            ]

        decoder += [
            nn.ConvTranspose2d(mid_channels[0], out_channel, kernel_size=kernel_size),
            nn.Sigmoid(),
        ]

        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        x = self.projection(x)
        x = x.view(-1, self.mid_channels[-1], self.feature_size, self.feature_size)
        x = self.decoder(x)

        return x


class ConvVAE(nn.Module):
    def __init__(
            self,
            in_channels,
            latent_dim,
            mid_channels,
            out_channels,
            img_size,
            kernel_size,
    ):
        super(ConvVAE, self).__init__()

        self.img_size = img_size

        self.encoder = Encoder(in_channels, latent_dim, mid_channels, img_size, kernel_size)
        self.decoder = Decoder(out_channels, latent_dim, mid_channels, self.encoder.img_size, kernel_size)

    def forward(self, x):
        # encode
        encoded = self.encoder(x)
        # reparameterize trick
        mu, log_var = encoded.chunk(2, dim=1)
        latent_z = self.reparameterize(mu, log_var)
        # decode
        decoded = self.decoder(latent_z)
        # reshape to image size
        decoded = F.interpolate(decoded, size=self.img_size, mode='bilinear', align_corners=False)

        return decoded, mu, log_var

    @staticmethod
    def reparameterize(mu, log_var):
        # compute standard deviation
        std = torch.exp(0.5 * log_var)
        # sample from standard normal distribution
        eps = torch.randn_like(std)
        # reparameterize the latent variable
        z = mu + eps * std

        return z
