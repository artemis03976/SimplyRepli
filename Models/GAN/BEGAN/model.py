import torch.nn as nn
import torch.nn.functional as F


class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel, use_conv=False, scale=2):
        super().__init__()

        self.scale = scale
        self.use_conv = use_conv

        if use_conv:
            self.conv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        if self.use_conv:
            x = self.conv(x)
        else:
            x = F.interpolate(x, scale_factor=self.scale, mode="nearest")

        return x


class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel, use_conv=True):
        super().__init__()

        if use_conv:
            self.down = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1)
        else:
            self.down = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.down(x)


class Generator(nn.Module):
    def __init__(
            self,
            latent_dim,
            base_channel,
            out_channel,
            num_layers,
            img_size,
    ):
        super(Generator, self).__init__()

        self.feature_size = img_size // 2 ** (num_layers - 1)

        self.generator_proj = nn.Linear(latent_dim, base_channel * self.feature_size * self.feature_size)

        self.generator = nn.ModuleList([])
        # decoder structure for generator
        for i in range(num_layers):
            self.generator.append(
                self.make_layer(base_channel, base_channel)
            )
            if i != num_layers - 1:
                self.generator.append(
                    Upsample(base_channel, base_channel)
                )

        self.generator.append(
            nn.Sequential(
                nn.Conv2d(base_channel, out_channel, kernel_size=3, stride=1, padding=1),
                # nn.Tanh(),
            )
        )

    @staticmethod
    def make_layer(in_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
        )

    def forward(self, x):
        x = self.generator_proj(x)
        # reshape to image shape
        x = x.view(x.shape[0], -1, self.feature_size, self.feature_size)

        for layer in self.generator:
            x = layer(x)

        return x


class Discriminator(nn.Module):
    def __init__(
            self,
            in_channel,
            base_channel,
            num_layers,
            latent_dim,
            img_size,
    ):
        super(Discriminator, self).__init__()

        self.feature_size = img_size // 2 ** (num_layers - 1)
        # encoder-decider structure for discriminator
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channel, base_channel, kernel_size=3, stride=1, padding=1),
                nn.ELU(),
            ),
        ])

        prev_channel = base_channel
        for i in range(num_layers):
            current_channel = (i + 1) * base_channel

            self.encoder.append(
                self.make_layer(prev_channel, current_channel)
            )
            if i != num_layers - 1:
                self.encoder.append(
                    Downsample(current_channel, current_channel)
                )

            prev_channel = current_channel

        # latent code projection
        self.encoder_proj = nn.Linear(self.feature_size * self.feature_size * num_layers * base_channel, latent_dim)
        self.decoder_proj = nn.Linear(latent_dim, self.feature_size * self.feature_size * base_channel)

        self.decoder = nn.ModuleList([])

        for i in range(num_layers):
            self.decoder.append(
                self.make_layer(base_channel, base_channel)
            )
            if i != num_layers - 1:
                self.decoder.append(
                    Upsample(base_channel, base_channel)
                )

        self.decoder.append(
            nn.Sequential(
                nn.Conv2d(base_channel, in_channel, kernel_size=3, stride=1, padding=1),
                # nn.Tanh(),
            )
        )

    @staticmethod
    def make_layer(in_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
        )

    def forward(self, x):
        for encoder_layer in self.encoder:
            x = encoder_layer(x)

        # reshape from flatten to image shape
        x = self.encoder_proj(x.view(x.shape[0], -1))
        x = self.decoder_proj(x).reshape(x.shape[0], -1, self.feature_size, self.feature_size)

        for decoder_layer in self.decoder:
            x = decoder_layer(x)

        return x
