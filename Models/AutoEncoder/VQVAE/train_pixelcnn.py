import torch
import torch.nn as nn
import torch.optim as optim

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
        total_loss = 0.0
        for batch_idx, (image, _) in enumerate(train_loader):
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

        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, total_loss / len(train_loader)))

    print("Finish training...")

    save_and_load.save_model(config, prior, network=config.prior_network)


def main():
    config_path = "config/config.yaml"
    config = VQVAEConfig(config_path)

    train_loader = load_data.load_train_data(config.prior_batch_size)

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
