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

    def make_layer(self, in_channel, out_channel, kernel_size=4, stride=1, padding=0):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
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

        self.discriminator = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channel, mid_channels[0], kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2),
            ),
        ])

        for i in range(len(mid_channels) - 1):
            self.discriminator.append(
                self.make_layer(mid_channels[i], mid_channels[i + 1], kernel_size=4, stride=2, padding=1)
            )

        self.discriminator.append(
            nn.Sequential(
                nn.Conv2d(mid_channels[-1], 1, kernel_size=4, stride=1, padding=0),
            )
        )

    def make_layer(self, in_channel, out_channel, kernel_size=4, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        for layer in self.discriminator:
            x = layer(x)

        return x
