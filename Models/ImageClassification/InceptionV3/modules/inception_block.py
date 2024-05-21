import torch
import torch.nn as nn

from Models.ImageClassification.InceptionV3.modules.basic_conv import BasicConv


class InceptionA(nn.Module):
    def __init__(self, in_channel, out_channel_pooling):
        super(InceptionA, self).__init__()

        # 1x1 branch
        self.branch_1 = BasicConv(in_channel, 64, kernel_size=1)

        # 5x5 branch
        self.branch_2 = nn.Sequential(
            BasicConv(in_channel, 48, kernel_size=1),
            BasicConv(48, 64, kernel_size=5, padding=2)
        )

        # 3x3 branch
        self.branch_3 = nn.Sequential(
            BasicConv(in_channel, 64, kernel_size=1),
            BasicConv(64, 96, kernel_size=3, padding=1),
            BasicConv(96, 96, kernel_size=3, padding=1)
        )

        # pooling branch
        self.branch_4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv(in_channel, out_channel_pooling, kernel_size=1)
        )

    def forward(self, x):
        branch_1 = self.branch_1(x)
        branch_2 = self.branch_2(x)
        branch_3 = self.branch_3(x)
        branch_4 = self.branch_4(x)

        outputs = [branch_1, branch_2, branch_3, branch_4]

        return torch.cat(outputs, dim=1)


class InceptionB(nn.Module):
    def __init__(self, in_channel, mid_channel):
        super(InceptionB, self).__init__()

        # 1x1 branch
        self.branch_1 = BasicConv(in_channel, 192, kernel_size=1)

        # 7x7 branch variants 1
        self.branch_2 = nn.Sequential(
            BasicConv(in_channel, mid_channel, kernel_size=1),
            BasicConv(mid_channel, mid_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv(mid_channel, 192, kernel_size=(7, 1), padding=(3, 0))
        )

        # 7x7 branch variants 2
        self.branch_3 = nn.Sequential(
            BasicConv(in_channel, mid_channel, kernel_size=1),
            BasicConv(mid_channel, mid_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv(mid_channel, mid_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv(mid_channel, mid_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv(mid_channel, 192, kernel_size=(7, 1), padding=(3, 0))
        )

        # pooling branch
        self.branch_4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv(in_channel, 192, kernel_size=1),
        )

    def forward(self, x):
        branch_1 = self.branch_1(x)
        branch_2 = self.branch_2(x)
        branch_3 = self.branch_3(x)
        branch_4 = self.branch_4(x)

        outputs = [branch_1, branch_2, branch_3, branch_4]

        return torch.cat(outputs, dim=1)


class InceptionC(nn.Module):
    def __init__(self, in_channel):
        super(InceptionC, self).__init__()

        # 1x1 branch
        self.branch_1 = BasicConv(in_channel, 320, kernel_size=1)

        # 3x3 branch variants
        self.branch_2 = BasicConv(in_channel, 384, kernel_size=1)
        self.branch_2a = BasicConv(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch_2b = BasicConv(384, 384, kernel_size=(3, 1), padding=(1, 0))

        # 5x5 branch variants
        self.branch_3 = nn.Sequential(
            BasicConv(in_channel, 448, kernel_size=1),
            BasicConv(448, 384, kernel_size=3, padding=1)
        )
        self.branch_3a = BasicConv(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch_3b = BasicConv(384, 384, kernel_size=(3, 1), padding=(1, 0))

        # pooling branch
        self.branch_4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv(in_channel, 192, kernel_size=1)
        )

    def forward(self, x):
        branch_1 = self.branch_1(x)

        branch_2 = self.branch_2(x)
        branch_2 = [
            self.branch_2a(branch_2),
            self.branch_2b(branch_2)
        ]
        branch_2 = torch.cat(branch_2, dim=1)

        branch_3 = self.branch_3(x)
        branch_3 = [
            self.branch_3a(branch_3),
            self.branch_3b(branch_3)
        ]
        branch_3 = torch.cat(branch_3, dim=1)

        branch_4 = self.branch_4(x)

        outputs = [branch_1, branch_2, branch_3, branch_4]

        return torch.cat(outputs, dim=1)
