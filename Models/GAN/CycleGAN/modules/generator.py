import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()

        self.res_block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_channel),
        )

        if in_channel != out_channel:
            self.skip_conn = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1)
        else:
            self.skip_conn = nn.Identity()

    def forward(self, x):
        residual = self.res_block(x)
        return self.skip_conn(x) + residual


class ResnetGenerator(nn.Module):
    def __init__(
            self,
            in_channel=3,
            out_channel=3,
            num_blocks=9,
            base_channel=64
    ):
        super(ResnetGenerator, self).__init__()

        self.init_block = nn.Sequential(
            nn.Conv2d(in_channel, base_channel, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.ReLU(),
        )

        self.down_blocks = nn.Sequential(
            self.make_layer(base_channel, base_channel * 2, kernel_size=3, down=True, stride=2, padding=1),
            self.make_layer(base_channel * 2, base_channel * 4, kernel_size=3, down=True, stride=2, padding=1)
        )

        self.res_blocks = nn.Sequential()
        for _ in range(num_blocks):
            self.res_blocks.append(ResBlock(base_channel * 4, base_channel * 4))

        self.up_blocks = nn.Sequential(
            self.make_layer(base_channel * 4, base_channel * 2, kernel_size=3, down=False, stride=2, padding=1, output_padding=1),
            self.make_layer(base_channel * 2, base_channel, kernel_size=3, down=False, stride=2, padding=1, output_padding=1)
        )

        self.final_block = nn.Sequential(
            nn.Conv2d(base_channel, out_channel, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.Tanh()
        )

    def make_layer(self, in_channel, out_channel, kernel_size, down=True, **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, **kwargs) if down
            else nn.ConvTranspose2d(in_channel, out_channel, kernel_size, **kwargs),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.init_block(x)

        x = self.down_blocks(x)
        x = self.res_blocks(x)
        x = self.up_blocks(x)

        x = self.final_block(x)

        return x







