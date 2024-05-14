import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from global_utilis import save_and_load
from Models.AutoEncoder.utilis import load_data
from config.config import VQVAEConfig
from model import VQVAE
from modules.pixelcnn import PixelCNN


def train(config, model, prior, train_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(prior.parameters(), lr=config.prior_learning_rate)
    num_epochs = config.prior_epochs

    print("Start training...")

    for epoch in range(num_epochs):
        # set progress bar
        train_info = tqdm(train_loader, unit="batch")
        train_info.set_description(f"Epoch {epoch + 1}/{num_epochs}")

        total_loss = train_step(model, prior, config, train_info, criterion, optimizer)

        print(
            'Epoch [{}/{}], Loss: {:.4f}'
            .format(epoch + 1, num_epochs, total_loss)
        )

    print("Finish training...")

    save_and_load.save_model(config, prior, network=config.prior_network)


def train_step(model, prior, config, train_info, criterion, optimizer):
    total_loss = 0.0
    for batch_idx, (image, _) in enumerate(train_info):
        with torch.no_grad():
            image = image.to(config.device)
            latent = model.encoder(image)
            latent = model.quantizer.quantize(latent).detach()

        logits = prior(latent)
        logits = logits.permute(0, 2, 3, 1).contiguous()

        loss = criterion(logits.view(-1, config.num_embeddings), latent.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        train_info.set_postfix(loss=loss.item())

    return total_loss / len(train_info)


def main():
    config_path = "config/config.yaml"
    config = VQVAEConfig(config_path)

    train_loader = load_data.get_train_loader(config)

    in_channel = out_channel = config.channel

    model = VQVAE(
        in_channel,
        out_channel,
        config.num_embeddings,
        config.embed_dim,
        config.num_res_blocks,
    ).to(config.device)

    save_and_load.load_weight(config, model)

    model.eval()

    prior = PixelCNN(
        config.num_embeddings,
        config.num_embeddings,
        config.mid_channel,
        config.num_res_blocks_prior,
    ).to(config.device)

    train(config, model, prior, train_loader)


if __name__ == "__main__":
    main()
