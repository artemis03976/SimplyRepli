import torch
import torch.nn as nn


# VAE encoder in linear
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims):
        super(Encoder, self).__init__()

        encoder = [
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LeakyReLU(0.2),
        ]

        for i in range(len(hidden_dims) - 1):
            encoder += [
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.LeakyReLU(0.2),
            ]

        encoder.append(nn.Linear(hidden_dims[-1], 2 * latent_dim))

        self.encoder = nn.Sequential(*encoder)

    def forward(self, x):
        return self.encoder(x)


# AE decoder in linear
class Decoder(nn.Module):
    def __init__(self, output_dim, latent_dim, hidden_dims):
        super(Decoder, self).__init__()

        decoder = [
            nn.Linear(latent_dim, hidden_dims[-1]),
            nn.LeakyReLU(0.2),
        ]

        for i in reversed(range(len(hidden_dims) - 1)):
            decoder += [
                nn.Linear(hidden_dims[i + 1], hidden_dims[i]),
                nn.LeakyReLU(0.2),
            ]

        decoder += [
            nn.Linear(hidden_dims[0], output_dim),
            nn.Sigmoid(),
        ]

        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        return self.decoder(x)


class LinearCVAE(nn.Module):
    def __init__(
            self,
            input_dim,
            latent_dim,
            hidden_dims,
            output_dim,
            num_classes,
    ):
        super(LinearCVAE, self).__init__()

        # VAE encoder
        self.encoder = Encoder(input_dim, latent_dim, hidden_dims)

        # VAE decoder
        self.decoder = Decoder(output_dim, latent_dim, hidden_dims)

        # label projection
        self.label_projection = nn.Sequential(
            nn.Linear(num_classes, latent_dim),
            nn.ReLU(),
        )

    def forward(self, x, y):
        # flatten the image
        batch_size, channel, img_height, img_width = x.shape
        x = x.view(batch_size, -1)
        # encode
        encoded = self.encoder(x)
        # reparameterize trick
        mu, log_var = encoded.chunk(2, dim=1)
        latent_z = self.reparameterize(mu, log_var)
        # add label embedding
        decoded = self.decoder(latent_z + self.label_projection(y.float()))
        # reshape to image size
        decoded = decoded.view(batch_size, channel, img_height, img_width)

        return decoded, mu, log_var

    def reparameterize(self, mu, log_var):
        # compute standard deviation
        std = torch.exp(0.5 * log_var)
        # sample from standard normal distribution
        eps = torch.randn_like(std)
        # reparameterize the latent variable
        z = mu + eps * std

        return z
