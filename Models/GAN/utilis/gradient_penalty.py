import torch
from torch.autograd import grad


def get_gradient_penalty(critic, real_data, fake_data, device):
    alpha = torch.rand(real_data.size(0), 1, 1, 1, device=device).expand_as(real_data)
    interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).requires_grad_(True)

    disc_interpolates = critic(interpolates)

    gradients = grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.shape).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)

    grad_penalty = torch.mean((torch.linalg.norm(gradients, dim=1) - 1) ** 2)

    return grad_penalty
