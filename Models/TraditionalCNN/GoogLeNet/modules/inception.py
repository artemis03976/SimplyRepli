import torch
import torch.nn as nn


class Inception(nn.Module):
    def __init__(self, in_channel, channel_1x1, channel_3x3, channel_5x5, pool_proj):
        super(Inception, self).__init__()
        self.branch_1 = nn.Conv2d(in_channel, channel_1x1, kernel_size=1)

        self.branch_2 = nn.Sequential(
            nn.Conv2d(in_channel, channel_3x3[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channel_3x3[0], channel_3x3[1], kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.branch_3 = nn.Sequential(
            nn.Conv2d(in_channel, channel_5x5[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channel_5x5[0], channel_5x5[1], kernel_size=5, padding=2),
            nn.ReLU(),
        )

        self.branch_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channel, pool_proj, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, x):
        branch_1 = self.branch_1(x)
        branch_2 = self.branch_2(x)
        branch_3 = self.branch_3(x)
        branch_4 = self.branch_4(x)

        return torch.cat([branch_1, branch_2, branch_3, branch_4], dim=1)


class InceptionAux(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(InceptionAux, self).__init__()

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(in_channel, 128, kernel_size=1),
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)
