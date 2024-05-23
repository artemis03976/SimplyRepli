import torch
import torch.nn as nn


class LinearGenerator(nn.Module):
    def __init__(
            self,
            latent_dim,
            hidden_dims,
            output_dim,
            num_classes,
            proj_dim,
    ):
        super(LinearGenerator, self).__init__()

        # label projection
        self.label_proj = nn.Linear(num_classes, proj_dim)

        self.generator = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim + proj_dim, hidden_dims[0]),
                nn.ReLU(),
            )
        ])

        for i in range(len(hidden_dims) - 1):
            self.generator.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.ReLU(),
                )
            )

        self.generator.append(
            nn.Sequential(
                nn.Linear(hidden_dims[-1], output_dim),
                nn.Tanh(),
            )
        )

    def forward(self, latent, label):
        label_embedding = self.label_proj(label)
        # concatenate label embedding and image tensor
        x = torch.cat([latent, label_embedding], dim=1)

        for layer in self.generator:
            x = layer(x)

        return x


class LinearDiscriminator(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dims,
            num_classes,
            proj_dim,
    ):
        super(LinearDiscriminator, self).__init__()

        # label projection
        self.label_proj = nn.Linear(num_classes, proj_dim)

        self.discriminator = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim + proj_dim, hidden_dims[0]),
                nn.LeakyReLU(0.2),
            )
        ])

        for i in range(len(hidden_dims) - 1):
            self.discriminator.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.LeakyReLU(0.2),
                )
            )

        self.discriminator.append(
            nn.Sequential(
                nn.Linear(hidden_dims[-1], 1),
                nn.Sigmoid(),
            )
        )

    def forward(self, image, label):
        label_embedding = self.label_proj(label)
        # concatenate label embedding and image tensor
        x = torch.cat([image, label_embedding], dim=1)

        for layer in self.discriminator:
            x = layer(x)
        return x
