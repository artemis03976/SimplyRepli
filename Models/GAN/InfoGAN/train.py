import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import itertools
from model import Generator, Discriminator
from config.config import InfoGANConfig
from Models.GAN.utilis import load_data
from global_utilis import save_and_load


def sample_noise(batch_size, config):
    # sample latent noise
    latent_noise = torch.randn(batch_size, config.latent_noise_dim, device=config.device)
    # sample latent discrete
    latent_discrete_idx = torch.randint(0, config.latent_discrete_dim, (batch_size, config.num_latent_discrete), device=config.device)
    latent_discrete = F.one_hot(latent_discrete_idx, config.latent_discrete_dim).float().view(batch_size, -1)
    # sample latent continuous
    latent_continuous = torch.rand(batch_size, config.latent_continuous_dim, device=config.device) * 2 - 1

    return latent_noise, latent_discrete, latent_discrete_idx, latent_continuous


def train(config, model, train_loader):
    generator, discriminator = model
    # pre-defined loss function and optimizer
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_discrete = nn.CrossEntropyLoss()

    optimizer_generator = optim.Adam(
        itertools.chain(generator.parameters(), discriminator.q_head.parameters()),
        lr=config.generator_lr,
        betas=(0.5, 0.999)
    )
    optimizer_discriminator = optim.Adam(
        itertools.chain(discriminator.main.parameters(), discriminator.d_head.parameters()),
        lr=config.discriminator_lr,
        betas=(0.5, 0.999)
    )

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
            (criterion_gan, criterion_discrete),
            (optimizer_generator, optimizer_discriminator)
        )

        print(
            '\nEpoch [{}/{}], Generator Loss: {:.4f}, Discriminator Loss: {:.4f}'
            .format(
                epoch + 1, num_epochs, total_loss[0], total_loss[1]
            )
        )

    print("Finish training...")

    save_and_load.save_model(config, model)


def train_step(model, config, train_info, criterion, optimizer):
    # unpacking
    generator, discriminator = model
    generator.train()
    discriminator.train()
    criterion_gan, criterion_discrete = criterion
    optimizer_generator, optimizer_discriminator = optimizer

    total_loss_g = 0.0
    total_loss_d = 0.0

    for batch_idx, (image, _) in enumerate(train_info):
        batch_size = image.shape[0]
        image = image.to(config.device)
        # labels for discriminator
        real_labels = torch.ones(batch_size, 1, device=config.device)
        fake_labels = torch.zeros(batch_size, 1, device=config.device)

        real_data = image
        # get fake data
        latent_noise, latent_discrete, latent_discrete_idx, latent_continuous = sample_noise(batch_size, config)
        latent = torch.cat([latent_noise, latent_discrete, latent_continuous], dim=1)
        fake_data = generator(latent)

        # step1: train discriminator
        for i in range(config.d_step):
            optimizer_discriminator.zero_grad()

            # calculate loss on real data
            output_real, _ = discriminator(real_data)
            loss_real = criterion_gan(output_real, real_labels)
            loss_real.backward()

            # calculate loss on fake data
            output_fake, _ = discriminator(fake_data.detach())
            loss_fake = criterion_gan(output_fake, fake_labels)
            loss_fake.backward()

            optimizer_discriminator.step()

            loss_discriminator = loss_real + loss_fake

            total_loss_d += loss_discriminator.item()

        # step2: train generator
        for i in range(config.g_step):
            optimizer_generator.zero_grad()

            d_output, q_output = discriminator(fake_data)
            loss_gan = criterion_gan(d_output, real_labels)

            # split output of latent code
            q_discrete, q_continuous = torch.split(
                q_output,
                [config.num_latent_discrete * config.latent_discrete_dim, config.latent_continuous_dim * 2],
                dim=1
            )

            # calculate loss for discrete latent code
            loss_discrete = 0.0
            for j in range(config.num_latent_discrete):
                current_latent_d = q_discrete[:, j * config.latent_discrete_dim: (j + 1) * config.latent_discrete_dim]
                loss_discrete += criterion_discrete(current_latent_d, latent_discrete_idx[:, j])

            # calculate loss for continuous latent code
            q_mu, q_var = q_continuous.chunk(2, dim=1)
            q_var = torch.exp(q_var)
            log_likelihood = -0.5 * torch.log(2 * np.pi * q_var + 1e-8) - (latent_continuous - q_mu) ** 2 / (2 * q_var + 1e-8)
            loss_continuous = -torch.mean(torch.sum(log_likelihood, dim=1))

            loss_generator = loss_gan + loss_discrete * config.lambda_discrete + loss_continuous * config.lambda_continuous
            loss_generator.backward()

            optimizer_generator.step()

            total_loss_g += loss_generator.item()
        # set progress bar info
        train_info.set_postfix(loss_d=loss_discriminator.item(), loss_g=loss_generator.item())

    return (
        total_loss_g / (len(train_info) * config.g_step),
        total_loss_d / (len(train_info) * config.d_step),
    )


def main():
    config_path = "config/config.yaml"
    config = InfoGANConfig(config_path)

    generator = Generator(
        config.latent_noise_dim + config.num_latent_discrete * config.latent_discrete_dim + config.latent_continuous_dim,
        config.channel,
        config.feature_size,
        config.base_channel,
        config.num_layers,
    ).to(config.device)

    discriminator = Discriminator(
        config.channel,
        config.feature_size,
        config.num_latent_discrete * config.latent_discrete_dim + config.latent_continuous_dim * 2,
        config.base_channel,
        config.num_layers,
    ).to(config.device)

    train_loader = load_data.get_train_loader(config)

    train(config, (generator, discriminator), train_loader)


if __name__ == '__main__':
    main()
