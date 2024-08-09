import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.UNet.TransUNet.modules.vision_transformer import VisionTransformer


class EncoderBottleneck(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, base_width=64):
        super(EncoderBottleneck, self).__init__()

        self.down_sample = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

        width = int(out_channel * (base_width / 64))

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, width, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, stride=2, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(width, out_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x_down = self.down_sample(x)

        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.conv_3(out)

        out = F.relu(x_down + out)

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, out_channel, num_heads, mlp_dim, num_layers, img_size, patch_size):
        super(Encoder, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

        self.bottleneck_1 = EncoderBottleneck(out_channel, out_channel * 2, stride=2)
        self.bottleneck_2 = EncoderBottleneck(out_channel * 2, out_channel * 4, stride=2)
        self.bottleneck_3 = EncoderBottleneck(out_channel * 4, out_channel * 8, stride=2)

        self.vit_img_size = img_size // patch_size

        self.vit = VisionTransformer(
            in_channel=out_channel * 8,
            patch_size=1,
            img_size=self.vit_img_size,
            num_encoders=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_channel * 8, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out_1 = self.conv_1(x)

        out_2 = self.bottleneck_1(out_1)
        out_3 = self.bottleneck_2(out_2)
        out = self.bottleneck_3(out_3)

        out = self.vit(out)
        out = out.transpose(1, 2).reshape(out.shape[0], -1, self.vit_img_size, self.vit_img_size)

        out = self.conv_2(out)

        return out, out_1, out_2, out_3
