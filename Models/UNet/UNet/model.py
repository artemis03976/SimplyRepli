import torch
import torch.nn as nn
from Models.UNet.utilis import crop


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class DownSample(nn.Module):
    def __init__(self):
        super(DownSample, self).__init__()

        self.down_sample = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        return self.down_sample(x)


class UpSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpSample, self).__init__()

        self.up_sample = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.up_sample(x)


class UNet(nn.Module):
    def __init__(
            self,
            in_channel,
            num_classes,
            ch_multi,
    ):
        super(UNet, self).__init__()

        self.base_channel = 64

        self.down_ch_list = [(in_channel, self.base_channel)] + [
            (self.base_channel * in_ch, self.base_channel * out_ch)
            for in_ch, out_ch in zip(ch_multi[:-1], ch_multi[1:])
        ]

        self.down_conv = nn.ModuleList([ConvBlock(in_ch, out_ch) for in_ch, out_ch in self.down_ch_list])

        self.downsample = nn.ModuleList([DownSample() for _ in range(len(self.down_ch_list))])

        self.mid_block = ConvBlock(self.base_channel * ch_multi[-1], self.base_channel * ch_multi[-1] * 2)

        self.up_ch_list = [(self.base_channel * ch_multi[-1] * 2, self.base_channel * ch_multi[-1])] + [
            (self.base_channel * in_ch, self.base_channel * out_ch)
            for in_ch, out_ch in zip(ch_multi[::-1], ch_multi[::-1][1:])
        ]

        self.up_conv = nn.ModuleList([ConvBlock(in_ch, out_ch) for in_ch, out_ch in self.up_ch_list])

        self.upsample = nn.ModuleList([UpSample(in_ch, out_ch) for in_ch, out_ch in self.up_ch_list])

        self.output_proj = nn.Conv2d(self.base_channel, num_classes, kernel_size=1)

    def forward(self, x):
        down_x_list = []

        for down_conv, downsample in zip(self.down_conv, self.downsample):
            x = down_conv(x)
            down_x_list.append(x)
            x = downsample(x)

        x = self.mid_block(x)

        for upsample, up_conv in zip(self.upsample, self.up_conv):
            x = upsample(x)
            down_x = down_x_list.pop()
            if down_x.shape != x.shape:
                down_x = crop.center_crop(x, down_x)
            x = torch.cat([x, down_x], dim=1)
            x = up_conv(x)

        x = self.output_proj(x)

        return x
