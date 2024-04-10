import torch.nn as nn


class Generator(nn.Module):
    def __init__(
            self,
            latent_dim,
            mid_channels,
            out_channel,
    ):
        super(Generator, self).__init__()

        self.generator = nn.ModuleList([
            self.make_layer(latent_dim, mid_channels[0], kernel_size=4, stride=1, padding=0)
        ])

        for i in range(len(mid_channels) - 1):
            self.generator.append(
                self.make_layer(mid_channels[i], mid_channels[i + 1], kernel_size=4, stride=2, padding=1)
            )

        self.generator.append(
            nn.Sequential(
                nn.ConvTranspose2d(mid_channels[-1], out_channel, kernel_size=4, stride=2, padding=1),
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
        if len(x.shape) != 4:
            x = x.unsqueeze(-1).unsqueeze(-1)

        for layer in self.generator:
            x = layer(x)

        return x


class Discriminator(nn.Module):
    def __init__(
            self,
            in_channel,
            mid_channels,
    ):
        super(Discriminator, self).__init__()

        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channel, mid_channels[0], kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2),
            ),
        ])

        for i in range(len(mid_channels) - 1):
            self.encoder.append(
                self.make_layer(mid_channels[i], mid_channels[i + 1], kernel_size=4, stride=2, padding=1)
            )

        self.decoder = nn.ModuleList([])

        for i in reversed(range(len(mid_channels) - 1)):
            self.decoder.append(
                self.make_layer(mid_channels[i + 1], mid_channels[i], down=False, kernel_size=4, stride=2, padding=1)
            )

        self.decoder.append(
            nn.ConvTranspose2d(mid_channels[0], in_channel, kernel_size=4, stride=2, padding=1)
        )

    @staticmethod
    def make_layer(in_channel, out_channel, down=True, **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, **kwargs) if down
            else nn.ConvTranspose2d(in_channel, out_channel, **kwargs),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        for encoder_layer in self.encoder:
            x = encoder_layer(x)

        latent_code = x

        for decoder_layer in self.decoder:
            x = decoder_layer(x)

        return x, latent_code
