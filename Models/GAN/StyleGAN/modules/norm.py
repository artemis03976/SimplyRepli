import torch
import torch.nn as nn

from Models.GAN.StyleGAN.modules.scaled_layer import ScaledLinear


class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()

        self.eps = 1e-8

    def forward(self, x):
        # return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, latent_dim, with_style=True):
        super(AdaptiveInstanceNorm, self).__init__()

        self.with_style = with_style

        self.instance_norm = nn.InstanceNorm2d(in_channel)
        if self.with_style:
            self.style = ScaledLinear(latent_dim, in_channel * 2, bias=False)

    def forward(self, x, style):
        x = self.instance_norm(x)

        if self.with_style:
            style = self.style(style).unsqueeze(2).unsqueeze(3)
            scale, shift = style.chunk(2, dim=1)
            x = x * (1 + scale) + shift

        return x
