import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from global_utilis import save_and_load
from Models.AutoEncoder.utilis import load_data
from config.config import AEConfig
from Models.AutoEncoder.AE.models.ae_linear import LinearAE
from Models.AutoEncoder.AE.models.ae_conv import ConvAE


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
    for batch_idx, (data, _) in enumerate(train_info):
        data = data.to(config.device)

        # forward propagation
        x_decoded = model(data)

        # compute loss
        loss = criterion(x_decoded, data)

        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        train_info.set_postfix(loss=loss.item())

    return total_loss / len(train_info)


def main():
    config_path = "config/config.yaml"
    config = AEConfig(config_path)

    train_loader = load_data.load_train_data(config.batch_size)

    if config.network == 'ae_linear':
        if isinstance(config.img_size, (tuple, list)):
            input_dim = output_dim = config.channel * config.img_size[0] * config.img_size[1]
        else:
            input_dim = output_dim = config.img_size ** 2

        model = LinearAE(
            input_dim,
            config.latent_dim_linear,
            config.hidden_dims,
            output_dim,
        ).to(config.device)

    elif config.network == 'ae_conv':
        in_channel = out_channel = config.channel

        model = ConvAE(
            in_channel,
            config.latent_dim_conv,
            config.mid_channels,
            out_channel,
            config.img_size,
            config.kernel_size
        ).to(config.device)

    else:
        raise NotImplementedError(f"Unsupported network: {config.network}")

    train(config, model, train_loader)


if __name__ == "__main__":
    main()
