import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()

        self.res_block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

        if in_channel != out_channel:
            self.skip_conn = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1)
        else:
            self.skip_conn = nn.Identity()

    def forward(self, x):
        residual = self.res_block(x)
        return self.skip_conn(x) + residual


class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()

        self.down_sample = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channel),
        )

    def forward(self, x):
        return self.down_sample(x)


class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channel),
        )

    def forward(self, x):
        return self.upsample(x)


class UNetBlock(nn.Module):
    def __init__(
            self,
            num_blocks,
            io_channel,
            mid_channel,
            mid_block,
            down_up_sample=True,
    ):
        super().__init__()
        self.down_block = nn.ModuleList([])
        self.up_block = nn.ModuleList([])

        for _ in range(num_blocks):
            self.down_block.append(ResBlock(io_channel, mid_channel))
            self.up_block.insert(0, ResBlock(mid_channel * 2, io_channel))
            io_channel = mid_channel

        if down_up_sample:
            self.downsample = Downsample(mid_channel, mid_channel)
            self.upsample = Upsample(mid_channel, mid_channel)
        else:
            self.downsample = nn.Identity()
            self.upsample = nn.Identity()

        self.mid_block = mid_block

    def forward(self, x):
        down = x
        for layer in self.down_block:
            down = layer(down)
        down_sampled = self.downsample(down)

        mid = self.mid_block(down_sampled)

        up = self.upsample(mid)
        up = torch.cat([down, up], dim=1)
        for layer in self.up_block:
            up = layer(up)

        return up


class UNetGenerator(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            num_blocks=1,
            base_channel=64,
            ch_mult=None,
    ):
        super().__init__()

        if ch_mult is None:
            ch_mult = [1, 2, 4, 8]

        self.in_projection = nn.Conv2d(in_channel, base_channel, kernel_size=1, stride=1)
        self.out_projection = nn.Conv2d(base_channel, out_channel, kernel_size=1, stride=1)

        dims = [base_channel] + [int(base_channel * mul) for mul in ch_mult]
        inout_channel = list(zip(dims[:-1], dims[1:]))

        unet_block = ResBlock(base_channel * ch_mult[-1], base_channel * ch_mult[-1])

        for io_channel, mid_channel in reversed(inout_channel):
            unet_block = UNetBlock(num_blocks, io_channel, mid_channel, unet_block)

        self.generator = unet_block

    def forward(self, x):
        x = self.in_projection(x)
        x = self.generator(x)
        x = self.out_projection(x)

        return x
