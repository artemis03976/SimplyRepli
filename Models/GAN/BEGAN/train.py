import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import Generator, Discriminator
from config.config import BEGANConfig
from Models.GAN.utilis import load_data
from global_utilis import save_and_load, plot


def train(config, model, train_loader):
    generator, discriminator = model

    criterion = nn.L1Loss()

    optimizer_generator = optim.Adam(generator.parameters(), lr=config.generator_lr, betas=(0.5, 0.999))
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=config.discriminator_lr, betas=(0.5, 0.999))

    num_epochs = config.epochs

    print("Start training...")

    for epoch in range(num_epochs):
        # set progress bar
        train_info = tqdm(train_loader, unit="batch")
        train_info.set_description(f"Epoch {epoch + 1}/{num_epochs}")

        total_loss = train_step(model, config, train_info, criterion,
                                (optimizer_generator, optimizer_discriminator))

        print(
            '\nEpoch [{}/{}], Generator Loss: {:.4f}, Discriminator Loss: {:.4f}'
            .format(
                epoch + 1, num_epochs, total_loss[0], total_loss[1]
            )
        )

        with torch.no_grad():
            samples = next(iter(train_loader))[0].to(config.device)
            recon = discriminator(samples)
            recon = 0.5 * (recon + 1)
            recon = recon.clamp(0, 1)
            plot.show_img(recon, cols=8)

            generation = generator(torch.rand(config.num_samples, config.latent_dim, device=config.device) * 2 - 1)
            generation = generation.view(config.num_samples, config.channel, config.img_size, config.img_size)
            gen = 0.5 * (generation + 1)
            gen = gen.clamp(0, 1)
            plot.show_img(gen, cols=8)

            recon = discriminator(generation)
            recon = 0.5 * (recon + 1)
            recon = recon.clamp(0, 1)
            plot.show_img(recon, cols=8)

    print("Finish training...")

    save_and_load.save_model(config, model)


def train_step(model, config, train_info, criterion, optimizer):
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
        z = torch.rand(batch_size, config.latent_dim, device=config.device) * 2 - 1
        fake_data = generator(z)

        # step1: train discriminator
        optimizer_discriminator.zero_grad()

        output_real = discriminator(real_data)
        loss_real = criterion(output_real, real_data)

        output_fake = discriminator(fake_data.detach())
        loss_fake = criterion(output_fake, fake_data.detach())

        loss_discriminator = loss_real - config.k * loss_fake
        loss_discriminator.backward()

        optimizer_discriminator.step()

        total_loss_d += loss_discriminator.item()

        # step2: train generator
        optimizer_generator.zero_grad()

        output_fake = discriminator(fake_data)
        loss_generator = criterion(output_fake, fake_data)
        loss_generator.backward()

        optimizer_generator.step()

        total_loss_g += loss_generator.item()

        # update k
        config.k += config.lambda_k * (config.gamma * loss_real.item() - loss_fake.item())
        config.k = max(min(config.k, 1.0), 0.0)

        measure = loss_real.item() + torch.abs(config.gamma * loss_real - loss_fake).item()

        train_info.set_postfix(loss_g=loss_generator.item(), loss_d=loss_discriminator.item(), measure=measure, k=config.k)

    return (
        total_loss_g / (len(train_info) * config.g_step),
        total_loss_d / (len(train_info) * config.d_step),
    )


def main():
    config_path = "config/config.yaml"
    config = BEGANConfig(config_path)

    in_channel = out_channel = config.channel

    generator = Generator(
        config.latent_dim,
        config.base_channel,
        out_channel,
        config.num_layers,
        config.img_size,
    ).to(config.device)

    discriminator = Discriminator(
        in_channel,
        config.base_channel,
        config.num_layers,
        config.latent_dim,
        config.img_size,
    ).to(config.device)

    train_loader = load_data.get_train_loader(config)

    train(config, (generator, discriminator), train_loader)


if __name__ == '__main__':
    main()
