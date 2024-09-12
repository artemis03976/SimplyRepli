import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from global_utilis import save_and_load
from Models.Diffusion.utilis import load_data
from config.config import CLIPConfig
from model import CLIP


def train(config, model, train_loader):
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.unet.parameters(), lr=config.learning_rate)

    num_epochs = config.epochs

    print("Start training...")

    for epoch in range(num_epochs):
        # set progress bar
        train_info = tqdm(train_loader, unit="batch")
        train_info.set_description(f"Epoch {epoch + 1}/{num_epochs}")

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
        ground_truth = torch.arange(data.shape[0])

        # forward propagation
        logits_per_image, logits_per_text = model(data)

        # compute loss
        loss = (criterion(logits_per_image, ground_truth) + criterion(logits_per_text, ground_truth)) / 2

        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        train_info.set_postfix(loss=loss.item())

    return total_loss / len(train_info)


def main():
    config_path = "config/config.yaml"
    config = CLIPConfig(config_path)

    train_loader = load_data.load_train_data(config)

    model = CLIP(
        config.align_dim,
        # vision part
        config.channel,
        config.patch_size,
        config.img_size,
        config.vision_num_layers,
        config.vision_num_heads,
        config.vision_mlp_dim,
        # text part
        config.vocab_size,
        config.text_max_len,
        config.text_embed_dim,
        config.text_num_layers,
        config.text_num_heads,
        config.text_mlp_dim,
    ).to(config.device)

    train(config, model, train_loader)


if __name__ == "__main__":
    main()
