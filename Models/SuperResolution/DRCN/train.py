import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from global_utilis import save_and_load
from Models.SuperResolution.utilis import load_data
from config.config import DRCNConfig
from model import DRCN
from loss import Loss


def train(config, model, train_loader):
    criterion = Loss(config.inference_depth, config.alpha, config.beta)
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
    criterion.decay_step()

    for batch_idx, (lr_img, sr_img) in enumerate(train_info):
        lr_img = lr_img.to(config.device)
        sr_img = sr_img.to(config.device)

        recon_sr_img, hidden = model(lr_img)

        loss = criterion(model, recon_sr_img, sr_img, hidden)

        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        train_info.set_postfix(loss=loss.item())

    return total_loss / len(train_info)


def main():
    config_path = "config/config.yaml"
    config = DRCNConfig(config_path)

    train_loader = load_data.get_train_loader(config)

    model = DRCN(
        config.channel,
        config.inference_depth,
    ).to(config.device)

    train(config, model, train_loader)


if __name__ == "__main__":
    main()
