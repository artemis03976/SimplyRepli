import torch
import torch.nn as nn
import torch.optim as optim

from global_utilis import save_and_load
from Models.AutoEncoder.utilis import load_data
from config.config import VAEConfig
from models.vae_linear import LinearVAE
from models.vae_conv import ConvVAE


def train(config, model, train_loader):
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    num_epochs = config.epochs

    print("Start training...")

    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, (data, _) in enumerate(train_loader):
            batch_size = data.shape[0]
            data = data.to(config.device)

            # forward propagation
            x_decoded, mu, log_var = model(data)

            # compute reconstruction loss
            reconstruction_loss = criterion(x_decoded.view(batch_size, -1), data.view(batch_size, -1))
            # compute KL divergence
            kl_divergence = torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1), dim=0)
            # add loss together
            loss = reconstruction_loss + kl_divergence

            # back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, total_loss / len(train_loader)))

    print("Finish training...")

    save_and_load.save_model(config, model)


def main():
    config_path = "config/config.yaml"
    config = VAEConfig(config_path)

    train_loader = load_data.load_train_data(config.batch_size)

    if config.network == 'vae_linear':
        if isinstance(config.img_size, (tuple, list)):
            input_dim = output_dim = config.channel * config.img_size[0] * config.img_size[1]
        else:
            input_dim = output_dim = config.img_size ** 2

        model = LinearVAE(
            input_dim,
            config.latent_dim_linear,
            config.hidden_dims,
            output_dim,
        ).to(config.device)

    elif config.network == 'vae_conv':
        in_channel = out_channel = config.channel

        model = ConvVAE(
            in_channel,
            config.latent_dim_conv,
            config.mid_channels,
            out_channel,
            config.img_size,
            config.kernel_size,
        ).to(config.device)

    else:
        raise NotImplementedError(f"Unsupported network: {config.network}")

    train(config, model, train_loader)


if __name__ == "__main__":
    main()
