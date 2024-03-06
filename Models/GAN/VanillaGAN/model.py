import torch.nn as nn


class Generator(nn.Module):
    def __init__(
            self,
            latent_dim,
            hidden_dims,
            output_dim,
    ):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        self.generator = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, hidden_dims[0]),
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

    def forward(self, x):
        for layer in self.generator:
            x = layer(x)
        return x


class Discriminator(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dims,
    ):
        super(Discriminator, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        self.discriminator = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0]),
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

    def forward(self, x):
        for layer in self.discriminator:
            x = layer(x)
        return x
