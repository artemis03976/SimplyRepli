import torch.nn as nn

from Models.ImageClassification.ResNet.modules.block import BuildingBlock, Bottleneck
from Models.ImageClassification.utilis.network import get_network_cfg


network_cfg = {
    'resnet18': ([2, 2, 2, 2], BuildingBlock),
    'resnet34': ([3, 4, 6, 3], BuildingBlock),
    'resnet50': ([3, 4, 6, 3], Bottleneck),
    'resnet101': ([3, 4, 23, 3], Bottleneck),
    'resnet152': ([3, 8, 36, 3], Bottleneck)
}


class ResNet(nn.Module):
    def __init__(
            self,
            in_channel,
            network,
            num_classes,
            init_weights=True,
    ):
        super(ResNet, self).__init__()

        self.base_channel = 64
        # get predefined network architecture
        num_blocks, block = get_network_cfg(network_cfg, network)

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer_1 = self.make_layer(block, 64, num_blocks[0], stride=1)
        self.layer_2 = self.make_layer(block, 128, num_blocks[1], stride=2)
        self.layer_3 = self.make_layer(block, 256, num_blocks[2], stride=2)
        self.layer_4 = self.make_layer(block, 512, num_blocks[3], stride=2)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512 * block.expansion, num_classes)
        )

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

    def make_layer(self, block, channel, num_block, stride=1):
        downsample = None
        if stride != 1 or self.base_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.base_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion),
            )

        layers = [block(self.base_channel, channel, stride=stride, downsample=downsample)]
        self.base_channel = channel * block.expansion

        for _ in range(1, num_block):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_layer(x)

        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)

        x = self.classifier(x)

        return x
