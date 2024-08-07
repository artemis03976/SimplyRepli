import torch
import torch.nn as nn

from Models.ImageClassification.utilis.network import get_network_cfg
from Models.ImageClassification.ConvNeXt.modules.block import *

network_cfg = {
    'ConvNeXt-T': ([3, 3, 9, 3], [96, 192, 384, 768]),
    'ConvNeXt-S': ([3, 3, 27, 3], [96, 192, 384, 768]),
    'ConvNeXt-B': ([3, 3, 27, 3], [128, 256, 512, 1024]),
    'ConvNeXt-L': ([3, 3, 27, 3], [192, 384, 768, 1536]),
    'ConvNeXt-XL': ([3, 3, 27, 3], [256, 512, 1024, 2048])
}


class ConvNeXt(nn.Module):
    def __init__(
            self,
            in_channel,
            network,
            num_classes,
            scale_init_value=1e-6,
            init_weights=True
    ):
        super(ConvNeXt, self).__init__()

        # get predefined network architecture
        num_blocks, mid_channels = get_network_cfg(network_cfg, network)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channel, mid_channels[0], kernel_size=4, stride=4),
            ChannelLayerNorm(mid_channels[0], mode='channel_first')
        )

        self.feature_layers = nn.ModuleList([
            nn.Sequential(
                *[ConvNeXtBlock(mid_channels[i], scale_init_value=scale_init_value) for _ in range(num_blocks[i])],
                Downsample(mid_channels[i], mid_channels[i + 1]) if i != len(num_blocks) - 1 else nn.Identity()
            ) for i in range(len(num_blocks))
        ])

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(mid_channels[-1]),
            nn.Linear(mid_channels[-1], num_classes)
        )

        if init_weights:
            self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.stem(x)

        for layer in self.feature_layers:
            x = layer(x)
        x = self.classifier(x)

        return x
