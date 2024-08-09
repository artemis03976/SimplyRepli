import torch
import torch.nn as nn


class DecoderBottleneck(nn.Module):
    def __init__(self, in_channel, out_channel, scale_factor=2):
        super(DecoderBottleneck, self).__init__()

        self.up_sample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, x_concat=None):
        x_up = self.up_sample(x)

        if x_concat is not None:
            x_up = torch.cat([x_up, x_concat], dim=1)

        out = self.conv_1(x_up)
        out = self.conv_2(out)

        return out


class Decoder(nn.Module):
    def __init__(self, out_channel, num_classes):
        super(Decoder, self).__init__()

        self.bottleneck_1 = DecoderBottleneck(out_channel * 8, out_channel * 2)
        self.bottleneck_2 = DecoderBottleneck(out_channel * 4, out_channel)
        self.bottleneck_3 = DecoderBottleneck(out_channel * 2, out_channel // 2)
        self.bottleneck_4 = DecoderBottleneck(out_channel // 2, out_channel // 8)

        self.conv_1 = nn.Conv2d(out_channel // 8, num_classes, kernel_size=1)

    def forward(self, x, out_1, out_2, out_3):
        out = self.bottleneck_1(x, out_3)
        out = self.bottleneck_2(out, out_2)
        out = self.bottleneck_3(out, out_1)
        out = self.bottleneck_4(out)

        out = self.conv_1(out)

        return out
