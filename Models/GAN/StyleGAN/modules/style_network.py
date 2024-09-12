import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.GAN.StyleGAN.modules.norm import PixelNorm, AdaptiveInstanceNorm
from Models.GAN.StyleGAN.modules.scaled_layer import ScaledLinear, ScaledConv2d


class NoiseFusion(nn.Module):
    def __init__(self, in_channel):
        super(NoiseFusion, self).__init__()

        self.weight = nn.Parameter(torch.zeros(1, in_channel, 1, 1))

    def forward(self, x):
        noise = torch.randn((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device)
        return x + self.weight * noise


class MappingNetwork(nn.Module):
    def __init__(self, latent_dim, style_dim):
        super(MappingNetwork, self).__init__()

        self.norm = PixelNorm()

        mapping = [
            self.make_layer(latent_dim, style_dim) if i == 1
            else self.make_layer(style_dim, style_dim)
            for i in range(8)
        ]
        self.mapping = nn.Sequential(*mapping)

    @staticmethod
    def make_layer(in_feature, out_feature, bias=True, weight_scale=True):
        return nn.Sequential(
            ScaledLinear(in_feature, out_feature, bias=bias, weight_scale=weight_scale),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.mapping(x)

        return x


class Upsample(nn.Module):
    def __init__(self, in_channel):
        super(Upsample, self).__init__()

        self.conv = ScaledConv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)

        return x


class SynthesizerBlock(nn.Module):
    def __init__(self, in_channel, out_channel, style_dim, upsample=True):
        super(SynthesizerBlock, self).__init__()

        if upsample:
            self.upsample = Upsample(in_channel)
        else:
            self.upsample = None

        self.conv_1 = ScaledConv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.noise_1 = NoiseFusion(out_channel)
        self.norm_1 = AdaptiveInstanceNorm(out_channel, style_dim)

        self.conv_2 = ScaledConv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.noise_2 = NoiseFusion(out_channel)
        self.norm_2 = AdaptiveInstanceNorm(out_channel, style_dim)

        self.act_func = nn.LeakyReLU(0.2)

    def forward(self, x, style):
        assert isinstance(style, tuple) and len(style) == 2
        if self.upsample is not None:
            x = self.upsample(x)

        x = self.conv_1(x)
        x = self.noise_1(x)
        x = self.act_func(x)
        x = self.norm_1(x, style[0])

        x = self.conv_2(x)
        x = self.noise_2(x)
        x = self.act_func(x)
        x = self.norm_2(x, style[1])

        return x


class Synthesizer(nn.Module):
    def __init__(
            self,
            in_channel,
            style_dim,
            img_size,
            max_channel=512,
    ):
        super(Synthesizer, self).__init__()

        self.constant_init = nn.Parameter(torch.ones(1, in_channel, 4, 4))
        self.init_noise_1 = NoiseFusion(in_channel)
        self.init_norm_1 = AdaptiveInstanceNorm(in_channel, style_dim)

        self.init_conv_2 = ScaledConv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.init_noise_2 = NoiseFusion(in_channel)
        self.init_norm_2 = AdaptiveInstanceNorm(in_channel, style_dim)

        self.act_func = nn.LeakyReLU(0.2)

        self.num_blocks = int(math.log2(img_size)) - 2
        self.mid_channels = list(reversed([min(2 ** i, max_channel) for i in range(4, self.num_blocks + 4)] + [in_channel]))

        self.synthesizer = nn.ModuleList([
            SynthesizerBlock(self.mid_channels[i], self.mid_channels[i + 1], style_dim, upsample=True)
            for i in range(self.num_blocks)
        ])

    def forward(self, style):
        x = self.constant_init.expand(style.shape[0], -1, -1, -1)
        x = self.init_noise_1(x)
        x = self.act_func(x)
        x = self.init_norm_1(x, style[:, 0])

        x = self.init_conv_2(x)
        x = self.init_noise_2(x)
        x = self.act_func(x)
        x = self.init_norm_2(x, style[:, 1])

        for idx, block in enumerate(self.synthesizer):
            style_for_block = (style[:, (idx + 1) * 2], style[:, (idx + 1) * 2 + 1])
            x = block(x, style_for_block)

        return x
