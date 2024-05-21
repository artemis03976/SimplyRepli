import torch
import torch.nn as nn


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, dropout):
        super(DenseLayer, self).__init__()

        self.bottleneck_expansion = 4

        self.conv_1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, self.bottleneck_expansion * growth_rate, kernel_size=1, stride=1, bias=False)
        )

        self.conv_2 = nn.Sequential(
            nn.BatchNorm2d(self.bottleneck_expansion * growth_rate),
            nn.ReLU(),
            nn.Conv2d(self.bottleneck_expansion * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # concat all previous features
        if isinstance(x, list):
            x = torch.cat(x, dim=1)

        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.dropout(x)

        return x


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, dropout):
        super(DenseBlock, self).__init__()

        self.dense_layer = nn.ModuleList([
            DenseLayer(in_channels + i * growth_rate, growth_rate, dropout) for i in range(num_layers)
        ])

    def forward(self, x):
        # store all previous features from dense layers
        total_features = [x]

        for layer in self.dense_layer:
            out = layer(total_features)
            # add new features to the list
            total_features.append(out)

        return torch.cat(total_features, dim=1)


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()

        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.transition(x)
