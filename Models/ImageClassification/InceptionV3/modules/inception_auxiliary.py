import torch
import torch.nn as nn

from Models.ImageClassification.InceptionV3.modules.basic_conv import BasicConv


class InceptionAux(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(InceptionAux, self).__init__()

        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3)

        self.conv_1 = BasicConv(in_channel, 128, kernel_size=1)
        self.conv_2 = BasicConv(128, 768, kernel_size=5)
        self.conv_2.stddev = 0.01

        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
