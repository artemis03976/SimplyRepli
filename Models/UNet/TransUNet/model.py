import torch
import torch.nn as nn
from modules.encoder import Encoder
from modules.decoder import Decoder


class TransUNet(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            num_classes,
            img_size,
            patch_size,
            num_layers,
            num_heads,
            mlp_dim
    ):
        super(TransUNet, self).__init__()

        self.encoder = Encoder(in_channel, out_channel, num_heads, mlp_dim, num_layers, img_size, patch_size)

        self.decoder = Decoder(out_channel, num_classes)

    def forward(self, x):
        out, out_1, out_2, out_3 = self.encoder(x)
        out = self.decoder(out, out_1, out_2, out_3)

        return out
