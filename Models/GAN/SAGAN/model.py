import torch
import torch.nn as nn
from modules.attention import AttentionBlock
from Models.GAN.utilis.spectral_norm import SpectralNorm
from modules.conditional_bn import ConditionalBatchNorm2d
from torch.nn.utils import spectral_norm


class Generator(nn.Module):
    def __init__(
            self,
            latent_dim,
            mid_channels,
            out_channel,
            img_size,
            num_classes
    ):
        super(Generator, self).__init__()

        self.generator = nn.ModuleList([
            self.make_layer(latent_dim, mid_channels[0], num_classes, kernel_size=3, stride=1, padding=0) if img_size == 28
            else self.make_layer(latent_dim, mid_channels[0], num_classes, kernel_size=4, stride=1, padding=0),
        ])

        for i in range(len(mid_channels) - 1):
            if i == 0 and img_size == 28:
                self.generator.append(
                    self.make_layer(mid_channels[i], mid_channels[i + 1], num_classes, kernel_size=3, stride=2, padding=0)
                )
            else:
                self.generator.append(
                    self.make_layer(mid_channels[i], mid_channels[i + 1], num_classes, kernel_size=4, stride=2, padding=1)
                )

        self.attn_block = AttentionBlock(mid_channels[-1])

        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(mid_channels[-1], out_channel, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    @staticmethod
    def make_layer(in_channel, out_channel, num_classes, **kwargs):
        return nn.ModuleList([
            SpectralNorm(nn.ConvTranspose2d(in_channel, out_channel, **kwargs)),
            ConditionalBatchNorm2d(out_channel, num_classes),
            nn.ReLU(),
        ])

    def forward(self, x, labels):
        if len(x.shape) != 4:
            x = x.unsqueeze(-1).unsqueeze(-1)

        # use ModuleList due to multiple input of ConditionalBatchNorm2d
        for conv, bn, act in self.generator:
            x = conv(x)
            x = bn(x, labels)
            x = act(x)

        x = self.attn_block(x)

        x = self.final_conv(x)

        return x


class Discriminator(nn.Module):
    def __init__(
            self,
            in_channel,
            mid_channels,
            img_size,
            num_classes,
    ):
        super(Discriminator, self).__init__()

        self.discriminator = nn.ModuleList([
            nn.Sequential(
                SpectralNorm(nn.Conv2d(in_channel, mid_channels[0], kernel_size=4, stride=2, padding=1)),
                nn.LeakyReLU(0.1),
            ),
        ])

        for i in range(len(mid_channels) - 1):
            self.discriminator.append(
                self.make_layer(mid_channels[i], mid_channels[i + 1], kernel_size=4, stride=2, padding=1)
            )

        self.attn_block = AttentionBlock(mid_channels[-1])

        self.final_conv = nn.Sequential(
            nn.Conv2d(mid_channels[-1], 1, kernel_size=4, stride=2, padding=1) if img_size == 28
            else nn.Conv2d(mid_channels[-1], 1, kernel_size=4, stride=1, padding=0),
        )

        self.label_projection = SpectralNorm(nn.Embedding(num_classes, mid_channels[-1]))

    @staticmethod
    def make_layer(in_channel, out_channel, **kwargs):
        return nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channel, out_channel, **kwargs)),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x, labels):
        for layer in self.discriminator:
            x = layer(x)

        x = self.attn_block(x)

        out_x = self.final_conv(x)
        out_x = out_x.view(-1, 1)

        label_proj = self.label_projection(labels)
        out_label = torch.sum(torch.sum(x, dim=[2, 3]) * label_proj, dim=1).view(-1, 1)

        return out_x + out_label
