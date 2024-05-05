import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.Diffusion.base_model.time_embedding import TimeEmbedding
from Models.Diffusion.base_model.attention_block import AttentionBlock


class ResBlock(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            time_embed_channel,
            use_attn=False
    ):
        super(ResBlock, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.GroupNorm(32, in_channel),
            nn.SiLU(),
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
        )

        self.conv_2 = nn.Sequential(
            nn.GroupNorm(32, out_channel),
            nn.SiLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
        )

        if in_channel != out_channel:
            self.skip_conn: nn.Module = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1)
        else:
            self.skip_conn: nn.Module = nn.Identity()

        if use_attn:
            self.attention_block = AttentionBlock(out_channel)
        else:
            self.attention_block = nn.Identity()

        self.time_embed_proj = nn.Linear(time_embed_channel, out_channel * 2)

    def forward(self, x, time_embed):
        h = self.conv_1(x)

        time_embed = self.time_embed_proj(time_embed)
        scale, shift = time_embed.unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        h = h * (scale + 1) + shift

        h = self.conv_2(h)

        h = h + self.skip_conn(x)

        h = self.attention_block(h)

        return h


class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel, use_conv=True, scale=2):
        super().__init__()

        self.scale = scale
        self.use_conv = use_conv

        if use_conv:
            self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor):
        x = x.float()
        x = F.interpolate(x, scale_factor=self.scale, mode="nearest")

        if self.use_conv:
            x = self.conv(x)

        return x


class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel, use_conv=True):
        super().__init__()

        if use_conv:
            self.down = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1)
        else:
            self.down = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor):
        return self.down(x)


class UNetBlock(nn.Module):
    def __init__(
            self,
            num_blocks,
            io_channel,
            mid_channel,
            time_embed_channel,
            mid_block,
            down_up_sample=True,
    ):
        super().__init__()
        self.down_block = nn.ModuleList([])
        self.up_block = nn.ModuleList([])

        for _ in range(num_blocks):
            self.down_block.append(ResBlock(io_channel, mid_channel, time_embed_channel))
            self.up_block.insert(0, ResBlock(mid_channel * 2, io_channel, time_embed_channel))
            io_channel = mid_channel

        if down_up_sample:
            self.downsample = Downsample(mid_channel, mid_channel)
            self.upsample = Upsample(mid_channel, mid_channel)
        else:
            self.downsample = nn.Identity()
            self.upsample = nn.Identity()

        self.mid_block = mid_block

    def forward(self, x, time_embed):
        down_stack = []
        down = x
        for layer in self.down_block:
            down = layer(down, time_embed)
            down_stack.append(down)

        down_sampled = self.downsample(down)
        mid = self.mid_block(down_sampled, time_embed)
        up = self.upsample(mid)

        for layer in self.up_block:
            down = down_stack.pop()
            up = torch.cat([down, up], dim=1)
            up = layer(up, time_embed)

        return up


class UNet(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            num_res_blocks=2,
            base_channel=64,
            time_embed_channel=None,
            ch_mult=None,
    ):
        super().__init__()

        if ch_mult is None:
            ch_mult = [1, 2, 4, 8]

        if time_embed_channel is None:
            time_embed_channel = base_channel * 4

        self.time_embedding = TimeEmbedding(time_embed_channel)

        self.in_projection = nn.Conv2d(in_channel, base_channel, kernel_size=3, stride=1, padding=1)
        self.out_projection = nn.Sequential(
            nn.GroupNorm(32, base_channel),
            nn.SiLU(),
            nn.Conv2d(base_channel, out_channel, kernel_size=3, stride=1, padding=1)
        )

        channels = [base_channel] + [int(base_channel * mul) for mul in ch_mult]
        inout_channel = list(zip(channels[:-1], channels[1:]))

        unet_block = ResBlock(base_channel * ch_mult[-1], base_channel * ch_mult[-1], time_embed_channel)

        for i, (io_channel, mid_channel) in enumerate(reversed(inout_channel)):
            unet_block = UNetBlock(
                num_res_blocks,
                io_channel,
                mid_channel,
                time_embed_channel,
                unet_block,
                down_up_sample=False if i == 0 else True
            )

        self.main = unet_block

    def forward(self, x, time_step):
        x = self.in_projection(x)

        time_embed = self.time_embedding(time_step)

        x = self.main(x, time_embed)

        x = self.out_projection(x)

        return x
