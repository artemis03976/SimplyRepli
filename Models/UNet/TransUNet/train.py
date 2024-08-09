import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config.config import TransUNetConfig
from model import TransUNet
from loss import Loss
from Models.UNet.utilis import crop, load_data
from global_utilis.early_stopping import EarlyStopping
from global_utilis import save_and_load


def train(config, model):
    criterion = Loss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    num_epochs = config.epochs

    train_loader, val_loader = load_data.get_train_val_loader(config)

    early_stopping = EarlyStopping(patience=3, delta=0.001, mode='min')

    print("Start training...")

    for epoch in range(num_epochs):
        # set progress bar
        train_info = tqdm(train_loader, unit="batch")
        train_info.set_description(f"Epoch {epoch + 1}/{num_epochs}")

        total_loss = train_step(model, config, train_info, criterion, optimizer)

        val_loss = validation(model, config, val_loader, criterion)

        print('\nEpoch [{}/{}], Average Loss: {:.4f}'.format(epoch + 1, num_epochs, total_loss / len(train_loader)))
        print('Validation Loss: {:.4f}'.format(val_loss))

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Need Early Stopping")
            break

    print("Finish training...")

    save_and_load.save_model(config, model)


def train_step(model, config, train_info, criterion, optimizer):
    model.train()
    total_loss = 0.0

    for batch_idx, (image, mask) in enumerate(train_info):
        image = image.to(config.device)
        mask = mask.to(config.device)

        prediction = model(image)

        mask = crop.center_crop(prediction, mask)

        loss = criterion(prediction, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        train_info.set_postfix(Loss=loss.item())

    return total_loss


def validation(model, config, val_loader, criterion):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_idx, (image, mask) in enumerate(val_loader):
            image = image.to(config.device)
            mask = mask.to(config.device)

            prediction = model(image)

            mask = crop.center_crop(prediction, mask)

            loss = criterion(prediction, mask)

            total_loss += loss.item()

    return total_loss / len(val_loader)


def main():
    config_path = "config/config.yaml"
    config = TransUNetConfig(config_path)

    model = TransUNet(
        config.channel,
        config.out_channel,
        config.num_classes,
        config.img_size,
        config.patch_size,
        config.num_layers,
        config.num_heads,
        config.mlp_dim
    ).to(config.device)

    train(config, model)


if __name__ == '__main__':
    main()
