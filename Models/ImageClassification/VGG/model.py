import torch
import torch.nn as nn


def get_network_cfg(network):
    if network == 'vgg11':
        num_blocks = [1, 1, 2, 2, 2]

    elif network == 'vgg13':
        num_blocks = [2, 2, 2, 2, 2]

    elif network == 'vgg16':
        num_blocks = [2, 2, 3, 3, 3]

    elif network == 'vgg19':
        num_blocks = [2, 2, 4, 4, 4]

    else:
        raise NotImplementedError(f'Unsupported model: {network}')

    return num_blocks


class VGG(nn.Module):
    def __init__(
            self,
            network,
            num_classes,
            dropout,
            init_weights=True,
    ):
        super(VGG, self).__init__()

        num_blocks = get_network_cfg(network)

        self.feature_layer = nn.ModuleList([])
        base_channel = 64

        for i in range(len(num_blocks)):
            if i == 0:
                self.feature_layer.append(self.make_layer(num_blocks[i], 3, base_channel))
            else:
                self.feature_layer.append(self.make_layer(num_blocks[i], base_channel, min(base_channel * 2, 512)))
                base_channel = min(512, base_channel * 2)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4096, num_classes)
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

    def make_layer(self, num_layers, in_channel, out_channel):
        conv_layer = []
        for _ in range(num_layers):
            conv_layer.append(nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1))
            in_channel = out_channel

        conv_layer.append(nn.ReLU())
        conv_layer.append(nn.MaxPool2d(kernel_size=2, stride=2))

        return nn.Sequential(*conv_layer)

    def forward(self, x):
        for layer in self.feature_layer:
            x = layer(x)

        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x
