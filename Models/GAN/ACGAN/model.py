import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(
            self,
            latent_dim,
            proj_dim,
            num_layers,
            out_channel,
    ):
        super(Generator, self).__init__()

        self.projection = nn.Linear(latent_dim, proj_dim)

        self.generator = nn.ModuleList([])

        for i in range(num_layers - 1):
            if i == 0:
                self.generator.append(
                    self.make_layer(proj_dim, proj_dim // 2, kernel_size=4, stride=1)
                )
            else:
                self.generator.append(
                    self.make_layer(proj_dim, proj_dim // 2, kernel_size=4, stride=2, padding=1)
                )
            proj_dim = proj_dim // 2

        self.generator.append(
            nn.Sequential(
                nn.ConvTranspose2d(proj_dim, out_channel, kernel_size=4, stride=2, padding=1),
                nn.Tanh(),
            )
        )

    def make_layer(self, in_channel, out_channel, **kwarg):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, **kwarg),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.projection(x)

        x = x.unsqueeze(-1).unsqueeze(-1)

        for layer in self.generator:
            x = layer(x)

        return x


class Discriminator(nn.Module):
    def __init__(
            self,
            in_channel,
            num_layers,
            dropout,
            img_size,
            num_classes,
            base_channel=16,
    ):
        super(Discriminator, self).__init__()

        self.num_classes = num_classes

        self.discriminator = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channel, base_channel, kernel_size=3, stride=2, padding=1),
                nn.Dropout(dropout),
                nn.LeakyReLU(0.2),
            ),
        ])

        img_size = img_size // 2

        for i in range(num_layers - 1):
            if i % 2 == 1:
                self.discriminator.append(
                    self.make_layer(base_channel, base_channel * 2, dropout, kernel_size=3, stride=2, padding=1)
                )
                img_size = img_size // 2
            else:
                self.discriminator.append(
                    self.make_layer(base_channel, base_channel * 2, dropout, kernel_size=3, stride=1, padding=1)
                )
            base_channel *= 2

        self.classifier = nn.Linear(img_size * img_size * base_channel, 1 + num_classes)

    def make_layer(self, in_channel, out_channel, dropout, **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, **kwargs),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        for layer in self.discriminator:
            x = layer(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x_dis, x_aux = torch.split(x, [1, self.num_classes], dim=1)

        x_dis = F.sigmoid(x_dis)

        return x_dis, x_aux
