import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXtBlock(nn.Module):
    def __init__(self, channel, scale_init_value=1e-6):
        super(ConvNeXtBlock, self).__init__()

        self.depth_wise_conv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=7, stride=1, padding=3, groups=channel),
            ChannelLayerNorm(channel, mode='channel_first')
        )

        self.point_wise_conv_1 = nn.Sequential(
            nn.Conv2d(channel, 4 * channel, kernel_size=1),
            nn.GELU()
        )

        self.scale = nn.Parameter(scale_init_value * torch.ones(channel),
                                  requires_grad=True) if scale_init_value > 0 else None

        self.point_wise_conv_2 = nn.Conv2d(4 * channel, channel, kernel_size=1)

    def forward(self, x):
        identity = x
        x = self.depth_wise_conv(x)
        x = self.point_wise_conv_1(x)
        x = self.point_wise_conv_2(x)

        if self.scale is not None:
            x = x * self.scale[:, None, None]

        return x + identity


class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=2, stride=2),
            ChannelLayerNorm(out_channel, mode='channel_first')
        )

    def forward(self, x):
        return self.downsample(x)


class ChannelLayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-6, mode="channel_last"):
        super(ChannelLayerNorm, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.mode = mode

        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.x = nn.LayerNorm(1)

    def forward(self, x):
        if self.mode == "channel_last":
            return F.layer_norm(x, (self.num_features,), self.gamma, self.beta, self.eps)
        elif self.mode == "channel_first":
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True)
            return self.gamma[:, None, None] * (x - mean) / (std + self.eps) + self.beta[:, None, None]
