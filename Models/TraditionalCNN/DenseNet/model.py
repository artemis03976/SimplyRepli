import torch.nn as nn

from Models.TraditionalCNN.DenseNet.modules.block import DenseBlock, TransitionLayer


network_cfg = {
    'densenet121': [6, 12, 24, 16],
    'densenet161': [6, 12, 36, 24],
    'densenet169': [6, 12, 32, 32],
    'densenet201': [6, 12, 48, 32],
    'densenet264': [6, 12, 64, 48],
}


def get_network_cfg(network):
    if network in network_cfg.keys():
        return network_cfg[network]

    else:
        raise NotImplementedError('Unsupported model: {}'.format(network))


class DenseNet(nn.Module):
    def __init__(
            self,
            network,
            num_classes,
            growth_rate,
            base_channel=64,
            dropout=0,
            init_weights=True,
    ):
        super(DenseNet, self).__init__()

        num_blocks = get_network_cfg(network)

        self.features = nn.Sequential(
             nn.Conv2d(3, base_channel, kernel_size=7, stride=2, padding=3, bias=False),
             nn.BatchNorm2d(base_channel),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.mid_blocks = nn.ModuleList([])

        for i in range(len(num_blocks)):
            self.mid_blocks.append(DenseBlock(num_blocks[i], base_channel, growth_rate, dropout))
            base_channel += num_blocks[i] * growth_rate

            if i != len(num_blocks) - 1:
                self.mid_blocks.append(TransitionLayer(base_channel, base_channel // 2))
                base_channel = base_channel // 2

        self.classifier = nn.Sequential(
            nn.BatchNorm2d(base_channel),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(base_channel, num_classes)
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
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        out = self.features(x)

        for layer in self.mid_blocks:
            out = layer(out)

        out = self.classifier(out)

        return out
