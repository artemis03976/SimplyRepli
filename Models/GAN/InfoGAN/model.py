import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(
            self,
            latent_dim,
            out_channel,
            feature_size,
            base_channel=64,
            num_layers=3,
            init_weight=True,
    ):
        super(Generator, self).__init__()

        self.feature_size = feature_size

        self.projection = nn.ModuleList([])

        if out_channel == 1:
            self.projection.append(
                nn.Sequential(
                    nn.Linear(latent_dim, 1024, bias=False),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                )
            )

        base_channel = min(256, base_channel * 2 ** (num_layers - 1)) if out_channel == 1 else 448
        flattened_dim = feature_size * feature_size * base_channel

        self.projection.append(
            nn.Sequential(
                nn.Linear(1024, flattened_dim, bias=False),
                nn.BatchNorm1d(flattened_dim),
                nn.ReLU(),
            )
        )

        self.generator = nn.Sequential()

        for i in range(num_layers):
            if i == num_layers - 1:
                self.generator.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(base_channel, out_channel, kernel_size=4, stride=2, padding=1, bias=False),
                        nn.Tanh() if out_channel == 3 else nn.Sigmoid()
                    )
                )
            else:
                self.generator.append(
                    self.make_layer(
                        base_channel,
                        256 if base_channel == 448 else base_channel // 2,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False
                    )
                )

                base_channel = 256 if base_channel == 448 else base_channel // 2

        if init_weight:
            self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    @staticmethod
    def make_layer(in_channel, out_channel, **kwargs):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, **kwargs),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

    def forward(self, x):
        for proj_layer in self.projection:
            x = proj_layer(x)

        x = x.view(x.size(0), -1, self.feature_size, self.feature_size)

        x = self.generator(x)

        return x


class Discriminator(nn.Module):
    def __init__(
            self,
            in_channel,
            feature_size,
            q_head_dim,
            base_channel=64,
            num_layers=3,
            init_weight=True,
    ):
        super(Discriminator, self).__init__()

        self.main = nn.ModuleList([])

        self.main.append(
            nn.Sequential(
                nn.Conv2d(in_channel, base_channel, kernel_size=4, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.1)
            )
        )

        for _ in range(num_layers - 1):
            self.main.append(
                self.make_layer(base_channel, base_channel * 2, kernel_size=4, stride=2, padding=1, bias=False)
            )
            base_channel *= 2

        if in_channel == 1:
            self.main.append(
                nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(feature_size * feature_size * base_channel, 1024, bias=False),
                    nn.BatchNorm1d(1024),
                    nn.LeakyReLU(0.1),
                )
            )

        # head for classification
        self.d_head = nn.Linear(1024 if in_channel == 1 else feature_size * feature_size * base_channel, 1)

        # head for latent representation
        self.q_head = nn.Sequential(
            nn.Linear(1024 if in_channel == 1 else feature_size * feature_size * base_channel, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, q_head_dim)
        )

        if init_weight:
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

    @staticmethod
    def make_layer(in_channel, out_channel, **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, **kwargs),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        for layer in self.main:
            x = layer(x)

        out_d = self.d_head(x)
        out_q = self.q_head(x)

        return out_d, out_q
