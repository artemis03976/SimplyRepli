import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config.config import UNetConfig
from model import UNet
from Models.UNet.utilis import crop, load_data
from global_utilis.early_stopping import EarlyStopping
from global_utilis import save_and_load


def train(config, model):
    criterion = nn.BCEWithLogitsLoss()
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

        val_loss = validation(model, config, val_loader)

        early_stopping(val_loss)

        print('\nEpoch [{}/{}], Average Loss: {:.4f}'.format(epoch + 1, num_epochs, total_loss / len(train_loader)))
        print('Validation Loss: {:.4f}'.format(val_loss))

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


def validation(model, config, val_loader):
    model.eval()
    total_loss = 0.0
    num_samples = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (image, mask) in enumerate(val_loader):
            image = image.to(config.device)
            mask = mask.to(config.device)

            prediction = model(image)

            mask = crop.center_crop(prediction, mask)

            loss = criterion(prediction, mask)

            total_loss += loss.item()
            num_samples += image.shape[0]

    return total_loss / num_samples


def main():
    config_path = "config/config.yaml"
    config = UNetConfig(config_path)

    model = UNet(
        config.channel,
        config.num_classes,
        config.ch_multi,
    ).to(config.device)

    train(config, model)


if __name__ == '__main__':
    main()
