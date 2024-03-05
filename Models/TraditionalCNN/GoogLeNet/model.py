import torch.nn as nn
from modules.inception import Inception, InceptionAux


class GoogLeNet(nn.Module):
    def __init__(
            self,
            num_classes,
            dropout=0.4,
            with_aux_logits=True,
            init_weights=True,
    ):
        super(GoogLeNet, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        )

        self.inception_3 = nn.Sequential(
            Inception(in_channel=192, channel_1x1=64, channel_3x3=[96, 128], channel_5x5=[16, 32], pool_proj=32),
            Inception(in_channel=256, channel_1x1=128, channel_3x3=[128, 192], channel_5x5=[32, 96], pool_proj=64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        )

        self.inception_4a = Inception(in_channel=480, channel_1x1=192, channel_3x3=[96, 208], channel_5x5=[16, 48],
                                      pool_proj=64)

        self.inception_4b = nn.Sequential(
            Inception(in_channel=512, channel_1x1=160, channel_3x3=[112, 224], channel_5x5=[24, 64], pool_proj=64),
            Inception(in_channel=512, channel_1x1=128, channel_3x3=[128, 256], channel_5x5=[24, 64], pool_proj=64),
            Inception(in_channel=512, channel_1x1=112, channel_3x3=[144, 288], channel_5x5=[32, 64], pool_proj=64, )
        )

        self.inception_4c = nn.Sequential(
            Inception(in_channel=528, channel_1x1=256, channel_3x3=[160, 320], channel_5x5=[32, 128], pool_proj=128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        )

        self.inception_5 = nn.Sequential(
            Inception(in_channel=832, channel_1x1=256, channel_3x3=[160, 320], channel_5x5=[32, 128], pool_proj=128),
            Inception(in_channel=832, channel_1x1=384, channel_3x3=[192, 384], channel_5x5=[48, 128], pool_proj=128),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(1024, num_classes),
        )

        if with_aux_logits:
            self.aux_1 = InceptionAux(512, num_classes)
            self.aux_2 = InceptionAux(528, num_classes)

        if init_weights:
            self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)

        x = self.inception_3(x)

        x = self.inception_4a(x)

        aux_1 = None
        if self.training and self.aux_1:
            aux_1 = self.aux_1(x)

        x = self.inception_4b(x)

        aux_2 = None
        if self.training and self.aux_2:
            aux_2 = self.aux_2(x)

        x = self.inception_4c(x)

        x = self.inception_5(x)
        x = self.classifier(x)

        return {
            'main': x,
            'aux_1': aux_1,
            'aux_2': aux_2
        }
