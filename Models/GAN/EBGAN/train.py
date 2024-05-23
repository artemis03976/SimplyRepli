import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import Generator, Discriminator
from config.config import EBGANConfig
from Models.GAN.utilis import load_data
from global_utilis import save_and_load


def train(config, model, train_loader):
    generator, discriminator = model
    # pre-defined loss function and optimizer
    criterion = nn.MSELoss()
    optimizer_generator = optim.Adam(generator.parameters(), lr=config.generator_lr, betas=(0.5, 0.999))
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=config.discriminator_lr, betas=(0.5, 0.999))

    num_epochs = config.epochs

    print("Start training...")

    for epoch in range(num_epochs):
        # set progress bar
        train_info = tqdm(train_loader, unit="batch")
        train_info.set_description(f"Epoch {epoch + 1}/{num_epochs}")

        # main train step
        total_loss = train_step(
            model,
            config,
            train_info,
            criterion,
            (optimizer_generator, optimizer_discriminator)
        )

        print(
            '\nEpoch [{}/{}], Generator Loss: {:.4f}, Discriminator Loss: {:.4f}'
            .format(epoch + 1, num_epochs, total_loss[0], total_loss[1])
        )

    print("Finish training...")

    save_and_load.save_model(config, model)


def pull_away_loss(latent_code):
    batch_size = latent_code.shape[0]
    latent_code = latent_code.view(batch_size, -1)

    norm = torch.norm(latent_code, dim=1)
    latent_code = latent_code / norm.unsqueeze(1)
    similarity = torch.matmul(latent_code, latent_code.transpose(0, 1))
    diag = torch.diag(similarity)
    pt = similarity - torch.diag(diag)
    loss_pt = (torch.sum(pt ** 2) / 2) / (batch_size * (batch_size - 1))

    return loss_pt


def train_step(model, config, train_info, criterion, optimizer):
    # unpacking
    generator, discriminator = model
    generator.train()
    discriminator.train()
    optimizer_generator, optimizer_discriminator = optimizer

    total_loss_g = 0.0
    total_loss_d = 0.0

    for batch_idx, (image, _) in enumerate(train_info):
        batch_size = image.shape[0]
        image = image.to(config.device)

        real_data = image
        # generate fake data
        z = torch.randn(batch_size, config.latent_dim, device=config.device)
        fake_data = generator(z)

        # step1: train discriminator
        for _ in range(config.d_step):
            optimizer_discriminator.zero_grad()

            # calculate loss on real data
            output_real, _ = discriminator(real_data)
            loss_real = criterion(output_real, real_data)

            # calculate loss on fake data
            output_fake, _ = discriminator(fake_data.detach())
            energy_fake = criterion(output_fake, fake_data.detach())
            loss_fake = torch.clamp(config.margin - energy_fake, min=0.0)
            loss_discriminator = loss_real + loss_fake
            loss_discriminator.backward()

            optimizer_discriminator.step()

            total_loss_d += loss_discriminator.item()

        # step2: train generator
        for _ in range(config.g_step):
            optimizer_generator.zero_grad()

            output_fake, latent_code = discriminator(fake_data)
            loss_gan = criterion(output_fake, fake_data)
            # calculate pull away loss
            loss_pt = config.lambda_pt * pull_away_loss(latent_code)
            loss_generator = loss_gan + loss_pt
            loss_generator.backward()

            optimizer_generator.step()

            total_loss_g += loss_generator.item()
        # set progress bar info
        train_info.set_postfix(loss_g=loss_generator.item(), loss_d=loss_discriminator.item())

    return (
        total_loss_g / (len(train_info) * config.g_step),
        total_loss_d / (len(train_info) * config.d_step),
    )


def main():
    config_path = "config/config.yaml"
    config = EBGANConfig(config_path)

    in_channel = out_channel = config.channel

    generator = Generator(
        config.latent_dim,
        config.G_mid_channels,
        in_channel,
        config.img_size,
    ).to(config.device)

    discriminator = Discriminator(
        out_channel,
        config.D_mid_channels,
    ).to(config.device)

    train_loader = load_data.get_train_loader(config)

    train(config, (generator, discriminator), train_loader)


if __name__ == '__main__':
    main()
