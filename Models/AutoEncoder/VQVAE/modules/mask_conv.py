import torch
import torch.nn as nn


class MaskConv2d(nn.Conv2d):
    def __init__(self, in_channel, out_channel, kernel_size, mask_type='A', **kwarg):
        super(MaskConv2d, self).__init__(
            in_channel, out_channel, kernel_size, **kwarg
        )

        self.mask_type = mask_type

        kernel_height, kernel_width = self.weight.size()[2:]

        mask = torch.ones(kernel_height, kernel_width)

        mask[kernel_height // 2 + 1:] = 0
        mask[kernel_height // 2, kernel_width // 2 + 1:] = 0

        if mask_type == 'A':
            mask[kernel_height // 2, kernel_width // 2] = 0

        mask = mask.unsqueeze(0).unsqueeze(0).repeat(out_channel, in_channel, 1, 1)

        self.register_buffer('mask', mask)

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskConv2d, self).forward(x)


class MaskedConvBlockA(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, **kwarg):
        super(MaskedConvBlockA, self).__init__()

        self.conv = nn.Sequential(
            MaskConv2d(in_channel, out_channel, kernel_size, mask_type='A', **kwarg),
            nn.BatchNorm2d(out_channel),
        )

    def forward(self, x):
        return self.conv(x)


class MaskedConvBlockB(nn.Module):
    def __init__(self, mid_channel, kernel_size, **kwarg):
        super(MaskedConvBlockB, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(2 * mid_channel, mid_channel, kernel_size=1),
            nn.BatchNorm2d(mid_channel),
        )

        self.mask_conv = nn.Sequential(
            nn.ReLU(),
            MaskConv2d(mid_channel, mid_channel, kernel_size, mask_type='B', **kwarg),
            nn.BatchNorm2d(mid_channel),
        )

        self.conv_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(mid_channel, 2 * mid_channel, kernel_size=1),
            nn.BatchNorm2d(2 * mid_channel),
        )

    def forward(self, x):
        out = self.conv_1(x)
        out = self.mask_conv(out)
        out = self.conv_2(out)

        return x + out
