import torch.nn as nn
from Models.AutoEncoder.VQVAE.modules.mask_conv import MaskedConvBlockA, MaskedConvBlockB


class PixelCNN(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            mid_channel,
            num_res_blocks,
    ):
        super(PixelCNN, self).__init__()

        # map discrete pixels to continuous latent variables
        self.embedding = nn.Embedding(in_channel, mid_channel)

        self.mask_conv_A = MaskedConvBlockA(mid_channel, 2 * mid_channel, kernel_size=7, stride=1, padding=3)

        self.mask_conv_B = nn.Sequential()
        for _ in range(num_res_blocks):
            self.mask_conv_B.append(
                MaskedConvBlockB(mid_channel, kernel_size=3, stride=1, padding=1)
            )

        self.out_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(2 * mid_channel, 1024, kernel_size=1, stride=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, out_channel, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x = self.embedding(x).permute(0, 3, 1, 2)

        x = self.mask_conv_A(x)
        x = self.mask_conv_B(x)
        x = self.out_conv(x)

        return x
