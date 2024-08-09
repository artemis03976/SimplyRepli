import torch
import torch.nn as nn


class FireBlock(nn.Module):
    def __init__(self, in_channel, squeeze_channel, expand1x1_channel, expand3x3_channel):
        super(FireBlock, self).__init__()

        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channel, squeeze_channel, kernel_size=1, stride=1),
            nn.ReLU(inplace=True)
        )

        self.expand1x1 = nn.Sequential(
            nn.Conv2d(squeeze_channel, expand1x1_channel, kernel_size=1, stride=1),
            nn.ReLU(inplace=True)
        )

        self.expand3x3 = nn.Sequential(
            nn.Conv2d(squeeze_channel, expand3x3_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        squeeze = self.squeeze(x)
        expand1x1 = self.expand1x1(squeeze)
        expand3x3 = self.expand3x3(squeeze)

        return torch.cat([expand1x1, expand3x3], dim=1)


class SqueezeNet(nn.Module):
    def __init__(
            self,
            in_channel,
            network,
            dropout=0.5,
            num_classes=1000,
            init_weights=True
    ):
        super(SqueezeNet, self).__init__()

        if network == 'v1.0':
            self.conv_1 = nn.Sequential(
                nn.Conv2d(in_channel, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
            )

            self.layer_2 = nn.Sequential(
                FireBlock(96, 16, 64, 64),
                FireBlock(128, 16, 64, 64),
                FireBlock(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
            )

            self.layer_3 = nn.Sequential(
                FireBlock(256, 32, 128, 128),
                FireBlock(256, 48, 192, 192),
                FireBlock(384, 48, 192, 192),
                FireBlock(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
            )

            self.layer_4 = nn.Sequential(
                FireBlock(512, 64, 256, 256)
            )

        elif network == 'v1.1':
            self.conv_1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
            )

            self.layer_2 = nn.Sequential(
                FireBlock(64, 16, 64, 64),
                FireBlock(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
            )

            self.layer_3 = nn.Sequential(
                FireBlock(128, 32, 128, 128),
                FireBlock(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
            )

            self.layer_4 = nn.Sequential(
                FireBlock(256, 48, 192, 192),
                FireBlock(384, 48, 192, 192),
                FireBlock(384, 64, 256, 256),
                FireBlock(512, 64, 256, 256)
            )

        else:
            raise NotImplementedError('Unsupported network: {}'.format(network))

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        if init_weights:
            self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.classifier(x)

        return torch.flatten(x, 1)
