import torch
import torch.nn as nn


class ConvGenerator(nn.Module):
    def __init__(
            self,
            latent_dim,
            mid_channels,
            out_channel,
            num_classes,
            proj_dim,
            img_size,
    ):
        super(ConvGenerator, self).__init__()

        self.latent_dim = latent_dim
        self.mid_channels = mid_channels
        self.out_channel = out_channel
        self.num_classes = num_classes
        self.proj_dim = proj_dim

        self.label_proj = nn.Linear(num_classes, proj_dim)

        self.generator = nn.ModuleList([
            self.make_layer(latent_dim + proj_dim, mid_channels[0], kernel_size=3, stride=1, padding=0) if img_size == 28
            else self.make_layer(latent_dim + proj_dim, mid_channels[0], kernel_size=4, stride=1, padding=0)
        ])

        for i in range(len(mid_channels) - 1):
            if i == 0 and img_size == 28:
                self.generator.append(
                    self.make_layer(mid_channels[i], mid_channels[i + 1], kernel_size=3, stride=2, padding=0)
                )
            else:
                self.generator.append(
                    self.make_layer(mid_channels[i], mid_channels[i + 1], kernel_size=4, stride=2, padding=1)
                )

        self.generator.append(
            nn.Sequential(
                nn.ConvTranspose2d(mid_channels[-1], out_channel, kernel_size=4, stride=2, padding=1),
                nn.Tanh(),
            )
        )

    @staticmethod
    def make_layer(in_channel, out_channel, **kwargs):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, **kwargs),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

    def forward(self, latent, label):
        if len(latent.shape) != 4:
            latent = latent.unsqueeze(-1).unsqueeze(-1)

        # concatenate label embedding and image tensor
        label_embedding = self.label_proj(label)
        label_embedding = label_embedding.unsqueeze(-1).unsqueeze(-1)
        x = torch.cat([latent, label_embedding], dim=1)

        for layer in self.generator:
            x = layer(x)

        return x


class ConvDiscriminator(nn.Module):
    def __init__(
            self,
            in_channel,
            mid_channels,
            num_classes,
            proj_dim,
            img_size,
    ):
        super(ConvDiscriminator, self).__init__()

        self.in_channel = in_channel
        self.mid_channels = mid_channels
        self.num_classes = num_classes
        self.proj_dim = proj_dim

        self.label_proj = nn.Linear(num_classes, proj_dim)

        self.discriminator = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channel + proj_dim, mid_channels[0], kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2),
            ),
        ])

        for i in range(len(mid_channels) - 1):
            self.discriminator.append(
                self.make_layer(mid_channels[i], mid_channels[i + 1], kernel_size=4, stride=2, padding=1)
            )

        self.discriminator.append(
            nn.Sequential(
                nn.Conv2d(mid_channels[-1], 1, kernel_size=4, stride=2, padding=1) if img_size == 28
                else nn.Conv2d(mid_channels[-1], 1, kernel_size=4, stride=1, padding=0),
                nn.Sigmoid(),
            )
        )

    @staticmethod
    def make_layer(in_channel, out_channel, **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, **kwargs),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2),
        )

    def forward(self, image, label):
        # concatenate label embedding and image tensor
        label_embedding = self.label_proj(label)
        # expand label embedding to match the size of image tensor
        label_embedding = label_embedding.unsqueeze(-1).unsqueeze(-1)
        label_embedding = label_embedding.expand(-1, -1, image.size(2), image.size(3))

        x = torch.cat([image, label_embedding], dim=1)

        for layer in self.discriminator:
            x = layer(x)

        x = x.view(-1, 1)

        return x
