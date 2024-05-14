import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from Models.AutoEncoder.utilis import load_data
from global_utilis import save_and_load
from config.config import CVAEConfig
from models.cvae_linear import LinearCVAE
from models.cvae_conv import ConvCVAE


def train(config, model, train_loader):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    num_epochs = config.epochs

    print("Start training...")

    for epoch in range(num_epochs):
        # set progress bar
        train_info = tqdm(train_loader, unit="batch")
        train_info.set_description(f"Epoch {epoch + 1}/{num_epochs}")

        total_loss = train_step(model, config, train_info, criterion, optimizer)

        print(
            'Epoch [{}/{}], Loss: {:.4f}'
            .format(epoch + 1, num_epochs, total_loss)
        )

    print("Finish training...")

    save_and_load.save_model(config, model)


def train_step(model, config, train_info, criterion, optimizer):
    total_loss = 0.0
    for batch_idx, (data, label) in enumerate(train_info):
        data = data.to(config.device)
        label = label.to(config.device)
        label = F.one_hot(label, config.num_classes)

        # forward propagation
        x_decoded, mu, log_var = model(data, label)

        # compute reconstruction loss
        reconstruction_loss = criterion(x_decoded, data)
        # compute KL divergence
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # add loss together
        loss = reconstruction_loss + kl_divergence

        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        train_info.set_postfix(loss=loss.item())

    return total_loss / len(train_info)


def main():
    config_path = "config/config.yaml"
    config = CVAEConfig(config_path)

    train_loader = load_data.get_train_loader(config)

    if config.network == 'cvae_linear':
        if isinstance(config.img_size, (tuple, list)):
            input_dim = output_dim = config.channel * config.img_size[0] * config.img_size[1]
        else:
            input_dim = output_dim = config.img_size ** 2

        model = LinearCVAE(
            input_dim,
            config.latent_dim_linear,
            config.hidden_dims,
            output_dim,
            config.num_classes,
        ).to(config.device)

    elif config.network == 'cvae_conv':
        in_channel = out_channel = config.channel

        model = ConvCVAE(
            in_channel,
            config.latent_dim_conv,
            config.mid_channels,
            out_channel,
            config.img_size,
            config.num_classes,
            config.kernel_size,
        ).to(config.device)

    else:
        raise NotImplementedError(f"Unsupported network: {config.network}")

    train(config, model, train_loader)


if __name__ == "__main__":
    main()
