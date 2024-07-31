import torch
import torch.nn as nn


class VDSR(nn.Module):
    def __init__(
            self,
            in_channel,
            num_layers,
            init_weights=True,
    ):
        super(VDSR, self).__init__()

        self.in_conv = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.res_blocks = nn.ModuleList([])
        for i in range(num_layers):
            self.res_blocks.append(
                self.make_layer(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
            )

        self.out_conv = nn.Conv2d(64, in_channel, kernel_size=3, stride=1, padding=1, bias=False)

        if init_weights:
            self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    @staticmethod
    def make_layer(in_channel, out_channel, **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, **kwargs),
            nn.ReLU()
        )

    def forward(self, x):
        identity = x

        x = self.in_conv(x)

        for res_block in self.res_blocks:
            x = res_block(x)

        x = self.out_conv(x)

        x = x + identity
        return x
