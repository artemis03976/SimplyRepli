import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.GAN.utilis import gradient_penalty


class GeneratorLoss(nn.Module):
    def __init__(self, config):
        super(GeneratorLoss, self).__init__()

        self.loss = config.loss
        self.grad_penalty_weight = config.grad_penalty_weight
        self.device = config.device

    def forward(self, discriminator, output_real, output_fake, real_data, fake_data):
        if self.loss == 'wgan_gp':
            # calculate gradient penalty
            grad_penalty = gradient_penalty.get_gradient_penalty(
                discriminator, real_data, fake_data, self.device
            )
            loss_discriminator = torch.mean(output_fake) - torch.mean(
                output_real) + self.grad_penalty_weight * grad_penalty
        elif self.loss == 'vanilla':
            loss_discriminator = torch.mean(F.softplus(output_fake)) + torch.mean(F.softplus(-output_real))
        else:
            raise NotImplementedError(f"Unsupported loss: {self.loss}")

        return loss_discriminator


class DiscriminatorLoss(nn.Module):
    def __init__(self, config):
        super(DiscriminatorLoss, self).__init__()

        self.loss = config.loss

    def forward(self, output_fake):
        if self.loss == 'wgan_gp':
            loss_generator = -torch.mean(output_fake)
        elif self.loss == 'vanilla':
            loss_generator = torch.mean(F.softplus(-output_fake))
        else:
            raise NotImplementedError(f"Unsupported loss: {self.loss}")

        return loss_generator
