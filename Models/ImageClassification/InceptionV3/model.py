import torch.nn as nn

from Models.ImageClassification.InceptionV3.modules.basic_conv import BasicConv
from Models.ImageClassification.InceptionV3.modules.inception_block import InceptionA, InceptionB, InceptionC
from Models.ImageClassification.InceptionV3.modules.inception_reduction_block import InceptionReductionA, InceptionReductionB
from Models.ImageClassification.InceptionV3.modules.inception_auxiliary import InceptionAux


class InceptionV3(nn.Module):
    def __init__(
            self,
            in_channel,
            num_classes,
            dropout=0.5,
            with_aux_logits=True,
            init_weights=True,
    ):
        super(InceptionV3, self).__init__()

        self.with_aux_logits = with_aux_logits

        self.conv_layer_1 = nn.Sequential(
            BasicConv(in_channel, 32, kernel_size=3, stride=2),
            BasicConv(32, 32, kernel_size=3, stride=1),
            BasicConv(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.conv_layer_2 = nn.Sequential(
            BasicConv(64, 80, kernel_size=1),
            BasicConv(80, 192, kernel_size=3),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.inception_1 = nn.Sequential(
            InceptionA(192, out_channel_pooling=32),
            InceptionA(256, out_channel_pooling=64),
            InceptionA(288, out_channel_pooling=64)
        )

        self.inception_2 = nn.Sequential(
            InceptionReductionA(288),
            InceptionB(768, mid_channel=128),
            InceptionB(768, mid_channel=160),
            InceptionB(768, mid_channel=160),
            InceptionB(768, mid_channel=192)
        )

        if with_aux_logits:
            self.aux_logits = InceptionAux(768, num_classes)

        self.inception_3 = nn.Sequential(
            InceptionReductionB(768),
            InceptionC(1280),
            InceptionC(2048),
        )

        self.main_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(2048, num_classes)
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
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)

        x = self.inception_1(x)
        x = self.inception_2(x)

        # aux branch
        aux = None
        if self.with_aux_logits and self.training:
            aux = self.aux_logits(x)

        x = self.inception_3(x)

        x = self.main_classifier(x)

        return {
            'main': x,
            'aux': aux,
        }
