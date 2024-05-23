import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from model import Generator, Discriminator
from config.config import ACGANConfig
from Models.GAN.utilis import load_data
from global_utilis import save_and_load


def train(config, model, train_loader):
    generator, discriminator = model
    # pre-defined loss function and optimizer
    criterion_dis = nn.BCELoss()
    criterion_aux = nn.CrossEntropyLoss()
    optimizer_generator = optim.Adam(generator.parameters(), lr=config.generator_lr)
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=config.discriminator_lr)

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
            (criterion_dis, criterion_aux),
            (optimizer_generator, optimizer_discriminator)
        )

        print(
            '\nEpoch [{}/{}], Generator Loss: {:.4f}, Discriminator Fake Loss: {:.4f}, Discriminator Real Loss: {:.4f}'
            .format(epoch + 1, num_epochs, total_loss[0], total_loss[1], total_loss[2])
        )

    print("Finish training...")

    save_and_load.save_model(config, model)


def train_step(model, config, train_info, criterion, optimizer):
    # unpacking
    generator, discriminator = model
    generator.train()
    discriminator.train()
    criterion_dis, criterion_aux = criterion
    optimizer_generator, optimizer_discriminator = optimizer

    total_loss_g = 0.0
    total_loss_d_fake = 0.0
    total_loss_d_real = 0.0

    for batch_idx, (image, label) in enumerate(train_info):
        batch_size = image.shape[0]
        image = image.to(config.device)
        label = label.to(config.device)
        # labels for discriminator
        real_labels = torch.ones(batch_size, 1, device=config.device)
        fake_labels = torch.zeros(batch_size, 1, device=config.device)

        # step1: train discriminator
        for _ in range(config.d_step):
            optimizer_discriminator.zero_grad()

            real_data = image
            # generate fake data and label
            z_img = torch.randn(batch_size, config.latent_dim, device=config.device)
            z_label = torch.randint(0, config.num_classes, (batch_size,), device=config.device)
            # one hot encoding for labels
            z_label_one_hot = F.one_hot(z_label, config.num_classes).to(z_img.dtype)
            fake_data = generator(torch.cat([z_img, z_label_one_hot], dim=1))

            # calculate loss on real data
            output_real_dis, output_real_aux = discriminator(real_data)
            loss_real_dis = criterion_dis(output_real_dis, real_labels)
            loss_real_aux = criterion_aux(output_real_aux, label)
            loss_real = loss_real_dis + loss_real_aux
            loss_real.backward()

            # calculate loss on fake data
            output_fake_dis, output_fake_aux = discriminator(fake_data.detach())
            loss_fake_dis = criterion_dis(output_fake_dis, fake_labels)
            loss_fake_aux = criterion_aux(output_fake_aux, z_label)
            loss_fake = loss_fake_dis + loss_fake_aux
            loss_fake.backward()

            optimizer_discriminator.step()

            total_loss_d_fake += loss_fake.item()
            total_loss_d_real += loss_real.item()

        # step2: train generator
        for _ in range(config.g_step):
            optimizer_generator.zero_grad()

            # generate fake data and label
            z_img = torch.randn(batch_size, config.latent_dim, device=config.device)
            z_label = torch.randint(0, config.num_classes, (batch_size,), device=config.device)
            # one hot encoding for labels
            z_label_one_hot = F.one_hot(z_label, config.num_classes).to(z_img.dtype)
            fake_data = generator(torch.cat([z_img, z_label_one_hot], dim=1))

            # loss with aux
            output_fake_dis, output_fake_aux = discriminator(fake_data)
            loss_generator_main = criterion_dis(output_fake_dis, real_labels)
            loss_generator_aux = criterion_aux(output_fake_aux, z_label)
            loss_generator = loss_generator_main + loss_generator_aux
            loss_generator.backward()

            optimizer_generator.step()

            total_loss_g += loss_generator.item()
        # set progress bar info
        train_info.set_postfix(loss_d_fake=loss_fake.item(), loss_d_real=loss_real.item(), loss_g=loss_generator.item())

    return (
        total_loss_g / (len(train_info) * config.g_step),
        total_loss_d_fake / (len(train_info) * config.d_step),
        total_loss_d_real / (len(train_info) * config.d_step)
    )


def main():
    config_path = "config/config.yaml"
    config = ACGANConfig(config_path)

    out_channel = in_channel = config.channel

    generator = Generator(
        config.latent_dim + config.num_classes,
        config.proj_dim,
        config.G_num_layers,
        out_channel,
        config.img_size
    ).to(config.device)

    discriminator = Discriminator(
        in_channel,
        config.D_num_layers,
        config.dropout,
        config.img_size,
        config.num_classes,
        config.base_channel,
    ).to(config.device)

    train_loader = load_data.get_train_loader(config)

    train(config, (generator, discriminator), train_loader)


if __name__ == '__main__':
    main()
