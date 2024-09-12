import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledLinear(nn.Module):
    def __init__(self, in_feature, out_feature, bias=True, weight_scale=True):
        super(ScaledLinear, self).__init__()

        if weight_scale:
            self.scale_factor = (2 / in_feature) ** 0.5
            self.init_factor = 1.0
        else:
            self.scale_factor = 1.0
            self.init_factor = (2 / in_feature) ** 0.5

        self.linear_weight = nn.Parameter(torch.randn(out_feature, in_feature) * self.init_factor)
        if bias:
            self.linear_bias = nn.Parameter(torch.zeros(out_feature))
        else:
            self.linear_bias = None

    def forward(self, x):
        return F.linear(x, self.linear_weight * self.scale_factor, self.linear_bias)


class ScaledConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, weight_scale=True, **kwargs):
        super(ScaledConv2d, self).__init__()

        self.kwargs = kwargs

        self.kernel_size = kwargs.get('kernel_size', 3)
        self.bias = kwargs.get('bias', True)
        self.stride = kwargs.get('stride', 1)
        self.padding = kwargs.get('padding', 0)
        self.dilation = kwargs.get('dilation', 1)
        self.groups = kwargs.get('groups', 1)

        if weight_scale:
            self.scale_factor = (2 / (in_channel * (self.kernel_size ** 2))) ** 0.5
            self.init_factor = 1.0
        else:
            self.scale_factor = 1.0
            self.init_factor = (2 / (in_channel * (self.kernel_size ** 2))) ** 0.5

        self.conv_weight = nn.Parameter(
            torch.randn(out_channel, in_channel, self.kernel_size, self.kernel_size) * self.init_factor
        )
        if self.bias:
            self.conv_bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.conv_bias = None

    def forward(self, x):
        return F.conv2d(
            x,
            self.conv_weight * self.scale_factor, self.conv_bias,
            self.stride, self.padding, self.dilation, self.groups
        )


if __name__ == '__main__':
    layer = ScaledConv2d(in_channel=3, out_channel=3, kernel_size=3, stride=1, padding=1)
    print(layer(torch.randn(1, 3, 32, 32)).shape)
