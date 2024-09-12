import math

import torch
import torch.nn as nn

from Models.GAN.StyleGAN.modules.style_network import MappingNetwork, Synthesizer, ScaledConv2d, ScaledLinear


class Generator(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            latent_dim,
            style_dim,
            img_size,
            style_avg_beta=None,
            style_mixing_prob=None,
            truncation_psi=None,
            truncation_cutoff=None
    ):
        super(Generator, self).__init__()

        self.style_mapping = MappingNetwork(latent_dim, style_dim)

        self.synthesizer = Synthesizer(in_channel, style_dim, img_size)

        self.num_blocks = self.synthesizer.num_blocks
        # num of style layers, every block has 2 adaIn style injection
        self.num_style_layers = len(self.synthesizer.mid_channels) * 2

        self.style_avg_beta = style_avg_beta
        self.style_avg = torch.zeros(style_dim)
        self.style_mixing_prob = style_mixing_prob
        self.truncation_psi = truncation_psi
        self.truncation_cutoff = truncation_cutoff

        self.to_rgb = ScaledConv2d(self.synthesizer.mid_channels[-1], out_channel, weight_scale=True, kernel_size=1)

    def moving_average(self, style):
        batch_avg = torch.mean(style[:, 0], dim=0)
        # Update style_avg using linear interpolation
        style_avg = self.style_avg * self.style_avg_beta + batch_avg * (1 - self.style_avg_beta)
        return style_avg

    def style_mix(self, style, latent_z):
        # Generate new latent vectors
        style_2 = torch.randn_like(latent_z)
        # Get new style through the mapping network
        style_2 = self.style_mapping(style_2).unsqueeze(1).expand(-1, self.num_style_layers, -1)

        # Determine mixing_cutoff
        if torch.rand(1).item() < self.style_mixing_prob:
            mixing_cutoff = torch.randint(1, self.num_style_layers, (1,)).item()
        else:
            mixing_cutoff = self.num_style_layers

        # Mix style and style_2 based on mixing_cutoff
        for layer_idx in range(self.num_style_layers):
            if layer_idx < mixing_cutoff:
                style[:, layer_idx] = style_2[:, layer_idx]

        return style

    def truncation(self, style):
        coefs = torch.ones(self.num_style_layers, dtype=torch.float32)

        for i in range(self.num_style_layers):
            if i < self.truncation_cutoff:
                coefs[i] *= self.truncation_psi

        coefs = coefs.unsqueeze(0).unsqueeze(-1)
        style_avg = self.style_avg.unsqueeze(0).unsqueeze(0)

        # Apply linear interpolation
        style = style_avg + (style - style_avg) * coefs

        return style

    def forward(self, latent_z):
        style = self.style_mapping(latent_z).unsqueeze(1).expand(-1, self.num_style_layers, -1)

        # apply moving average
        if self.style_avg_beta is not None:
            self.style_avg = self.moving_average(style)

        # apply style mixing
        if self.style_mixing_prob is not None:
            style = self.style_mix(style, latent_z)

        # apply truncation trick
        if self.truncation_psi is not None and self.truncation_cutoff is not None:
            style = self.truncation(style)

        # synthesize image with latent style vector
        img = self.synthesizer(style)

        img = self.to_rgb(img)

        return img


class Downsample(nn.Module):
    def __init__(self, in_channel, weight_scale=True):
        super(Downsample, self).__init__()

        self.conv = ScaledConv2d(in_channel, in_channel, weight_scale=weight_scale, kernel_size=3, stride=2, padding=1)
        self.act_func = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act_func(self.conv(x))


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channel, out_channel, weight_scale=True, downsample=True):
        super(DiscriminatorBlock, self).__init__()

        self.conv = ScaledConv2d(in_channel, out_channel, weight_scale=weight_scale, kernel_size=3, stride=1, padding=1)
        self.act_func = nn.LeakyReLU(0.2)

        if downsample:
            self.downsample = Downsample(out_channel)
        else:
            self.downsample = None

    def forward(self, x):
        x = self.act_func(self.conv(x))

        if self.downsample is not None:
            x = self.downsample(x)

        return x


class Discriminator(nn.Module):
    def __init__(
            self,
            in_channel,
            img_size,
            max_channel=512,
    ):
        super(Discriminator, self).__init__()

        self.num_blocks = int(math.log2(img_size)) - 2
        self.mid_channels = list([min(2 ** i, max_channel) for i in range(4, self.num_blocks + 5)])

        self.from_rgb = ScaledConv2d(in_channel, self.mid_channels[0], weight_scale=True, kernel_size=1)

        self.act_func = nn.LeakyReLU(0.2)

        self.discriminator = nn.ModuleList([
            DiscriminatorBlock(self.mid_channels[i], self.mid_channels[i + 1], downsample=True)
            for i in range(self.num_blocks)
        ])

        self.final = nn.Sequential(
            DiscriminatorBlock(self.mid_channels[-1], self.mid_channels[-1], downsample=False),
            nn.Flatten(),
            ScaledLinear(self.mid_channels[-1] * 4 * 4, self.mid_channels[-1], weight_scale=True),
            nn.LeakyReLU(0.2),
            ScaledLinear(self.mid_channels[-1], 1, weight_scale=True)
        )

    def forward(self, x):
        x = self.act_func(self.from_rgb(x))

        for layer in self.discriminator:
            x = layer(x)

        x = self.final(x)

        return x


if __name__ == '__main__':
    model = Generator(
        512, 3, 256, 256, 1024,
        style_avg_beta=0.995, style_mixing_prob=0.9, truncation_psi=0.9, truncation_cutoff=8
    )
    style = torch.randn(2, 256)
    img = model(style)
    print(img.shape)

    discriminator = Discriminator(
        3, 1024
    )

    print(discriminator(img).shape)
