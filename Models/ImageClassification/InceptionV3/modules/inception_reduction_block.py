import torch
import torch.nn as nn

from Models.TraditionalCNN.InceptionV3.modules.basic_conv import BasicConv


class InceptionReductionA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionReductionA, self).__init__()

        self.branch_1 = BasicConv(in_channels, 384, kernel_size=3, stride=2)

        self.branch_2 = nn.Sequential(
            BasicConv(in_channels, 64, kernel_size=1),
            BasicConv(64, 96, kernel_size=3, padding=1),
            BasicConv(96, 96, kernel_size=3, stride=2)
        )

        self.branch_3 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        branch_1 = self.branch_1(x)
        branch_2 = self.branch_2(x)
        branch_3 = self.branch_3(x)

        outputs = [branch_1, branch_2, branch_3]

        return torch.cat(outputs, dim=1)


class InceptionReductionB(nn.Module):
    def __init__(self, in_channels):
        super(InceptionReductionB, self).__init__()

        self.branch_1 = nn.Sequential(
            BasicConv(in_channels, 192, kernel_size=1),
            BasicConv(192, 320, kernel_size=3, stride=2)
        )

        self.branch_2 = nn.Sequential(
            BasicConv(in_channels, 192, kernel_size=1),
            BasicConv(192, 192, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv(192, 192, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv(192, 192, kernel_size=3, stride=2)
        )

        self.branch_3 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        branch_1 = self.branch_1(x)
        branch_2 = self.branch_2(x)
        branch_3 = self.branch_3(x)

        outputs = [branch_1, branch_2, branch_3]

        return torch.cat(outputs, dim=1)
