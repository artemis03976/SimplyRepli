import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from global_utilis import save_and_load
from Models.Diffusion.utilis import load_data
from config.config import DDPMConfig
from model import DDPM


def train(config, model, train_loader):
    # pre-defined loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.unet.parameters(), lr=config.learning_rate)

    num_epochs = config.epochs

    print("Start training...")

    for epoch in range(num_epochs):
        # set progress bar
        train_info = tqdm(train_loader, unit="batch")
        train_info.set_description(f"Epoch {epoch + 1}/{num_epochs}")

        # main train step
        total_loss = train_step(model, config, train_info, criterion, optimizer)

        print(
            '\nEpoch [{}/{}], Loss: {:.4f}'
            .format(
                epoch + 1, num_epochs, total_loss
            )
        )

    print("Finish training...")

    save_and_load.save_model(config, model)


def train_step(model, config, train_info, criterion, optimizer):
    total_loss = 0.0
    for batch_idx, (data, _) in enumerate(train_info):
        data = data.to(config.device)

        pred_noise, original_noise = model(data)

        loss = criterion(pred_noise, original_noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        # set progress bar info
        train_info.set_postfix(loss=loss.item())

    return total_loss / len(train_info)


def main():
    config_path = "config/config.yaml"
    config = DDPMConfig(config_path)
    # get train data loader
    train_loader = load_data.load_train_data(config)

    in_channel = out_channel = config.channel

    model = DDPM(
        in_channel,
        out_channel,
        config.num_res_blocks,
        config.base_channel,
        config.time_embed_channel,
        config.ch_mult,
        config.num_time_step,
        config.betas
    ).to(config.device)

    train(config, model, train_loader)


if __name__ == "__main__":
    main()
