import torch.nn as nn


class Generator(nn.Module):
    def __init__(
            self,
            latent_dim,
            mid_channels,
            out_channel,
            img_size,
    ):
        super(Generator, self).__init__()

        self.generator = nn.ModuleList([
            self.make_layer(latent_dim, mid_channels[0], kernel_size=3, stride=1, padding=0) if img_size == 28
            else self.make_layer(latent_dim, mid_channels[0], kernel_size=4, stride=1, padding=0),
        ])

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
        # protection for incorrect shape
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
            img_size,
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
                nn.Conv2d(mid_channels[-1], 1, kernel_size=4, stride=2, padding=1) if img_size == 28
                else nn.Conv2d(mid_channels[-1], 1, kernel_size=4, stride=1, padding=0),
            )
        )

    @staticmethod
    def make_layer(in_channel, out_channel, **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, **kwargs),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        for layer in self.discriminator:
            x = layer(x)

        return x
