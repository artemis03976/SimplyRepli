import torch.nn as nn
import torch.optim as optim

from global_utilis import save_and_load
from Models.AutoEncoder.utilis import load_data
from config.config import VQVAEConfig
from model import VQVAE


def train(config, model, train_loader):
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=config.model_learning_rate)
    num_epochs = config.model_epochs

    print("Start training...")

    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, (image, _) in enumerate(train_loader):
            batch_size = image.shape[0]
            image = image.to(config.device)

            # forward propagation
            x_recon, embedding_loss, commitment_loss = model(image)

            # compute reconstruction loss
            reconstruction_loss = criterion(x_recon.view(batch_size, -1), image.view(batch_size, -1))
            # add loss together
            loss = reconstruction_loss + embedding_loss + config.beta * commitment_loss

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
    config = VQVAEConfig(config_path)

    train_loader = load_data.load_train_data(config.model_batch_size)

    in_channel = out_channel = config.channel

    model = VQVAE(
        in_channel,
        out_channel,
        config.num_embeddings,
        config.embed_dim,
        config.num_res_blocks,
    ).to(config.device)

    train(config, model, train_loader)


if __name__ == "__main__":
    main()
