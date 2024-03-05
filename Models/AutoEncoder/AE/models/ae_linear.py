import torch
import torch.nn as nn


# AE encoder in linear
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims):
        super(Encoder, self).__init__()

        encoder = [
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
        ]

        for i in range(len(hidden_dims) - 1):
            encoder += [
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.ReLU(),
            ]

        encoder.append(nn.Linear(hidden_dims[-1], latent_dim))

        self.encoder = nn.Sequential(*encoder)

    def forward(self, x):
        return self.encoder(x)


# AE decoder in linear
class Decoder(nn.Module):
    def __init__(self, output_dim, latent_dim, hidden_dims):
        super(Decoder, self).__init__()

        decoder = [
            nn.Linear(latent_dim, hidden_dims[-1]),
            nn.ReLU(),
        ]

        for i in reversed(range(len(hidden_dims) - 1)):
            decoder += [
                nn.Linear(hidden_dims[i + 1], hidden_dims[i]),
                nn.ReLU(),
            ]

        decoder += [
            nn.Linear(hidden_dims[0], output_dim),
            nn.Sigmoid(),
        ]

        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        return self.decoder(x)


# AE in linear
class LinearAE(nn.Module):
    def __init__(
            self,
            input_dim,
            latent_dim,
            hidden_dims,
            output_dim,
    ):
        super(LinearAE, self).__init__()

        # AE encoder
        self.encoder = Encoder(input_dim, latent_dim, hidden_dims)

        # AE decoder
        self.decoder = Decoder(output_dim, latent_dim, hidden_dims)

    def forward(self, x):
        # flatten the image
        batch_size, channel, img_height, img_width = x.shape
        x = x.view(batch_size, -1)
        # encode
        encoded = self.encoder(x)
        # decode
        decoded = self.decoder(encoded)
        # reshape to image size
        decoded = decoded.view(batch_size, channel, img_height, img_width)

        return decoded
