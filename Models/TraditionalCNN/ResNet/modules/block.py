import torch.nn as nn
import torch.nn.functional as F


# module for ResNet18, ResNet34
class BuildingBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BuildingBlock, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )

        self.downsample = downsample

    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)

        x = self.conv_1(x)
        x = self.conv_2(x)
        x = F.relu(x + identity)

        return x


# module for ResNet50, ResNet101, ResNet152
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel * 4, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel * 4)
        )

        self.downsample = downsample

    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)

        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = F.relu(x + identity)

        return x
