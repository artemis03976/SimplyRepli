import torch.nn as nn
from modules.vector_quantizer import VectorQuantizer


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )

    def forward(self, x):
        return x + self.block(x)


class Encoder(nn.Module):
    def __init__(self, in_channel, num_res_blocks):
        super(Encoder, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.res_blocks = nn.Sequential()

        for _ in range(num_res_blocks):
            self.res_blocks.append(ResBlock(256, 256))

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.res_blocks(x)

        return x


class Decoder(nn.Module):
    def __init__(self, num_res_blocks, out_channel):
        super(Decoder, self).__init__()

        self.deconv_1 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.deconv_2 = nn.Sequential(
            nn.ConvTranspose2d(64, out_channel, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),
        )

        self.res_blocks = nn.Sequential()

        for _ in range(num_res_blocks):
            self.res_blocks.append(ResBlock(256, 256))

    def forward(self, x):
        x = self.res_blocks(x)
        x = self.deconv_1(x)
        x = self.deconv_2(x)

        return x


class VQVAE(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            num_embeddings,
            embed_dim,
            num_res_blocks,
    ):
        super(VQVAE, self).__init__()

        self.encoder = Encoder(in_channel, num_res_blocks)
        self.quantizer = VectorQuantizer(num_embeddings, embed_dim)
        self.decoder = Decoder(num_res_blocks, out_channel)

    def forward(self, x):
        # encode
        z_e = self.encoder(x)
        # quantize
        z_q, embedding_loss, commitment_loss = self.quantizer(z_e)
        # decode
        x_recon = self.decoder(z_q)

        return x_recon, embedding_loss, commitment_loss
