import torch
from torch.autograd import grad


def get_w_divergence(real_data, output_real, fake_data, output_fake, device, k=2, p=6):
    batch_size = real_data.shape[0]

    real_data = real_data.requires_grad_(True)

    real_grad = grad(
        outputs=output_real,
        inputs=real_data,
        grad_outputs=torch.ones(output_real.shape).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0].view(batch_size, -1)

    fake_grad = grad(
        outputs=output_fake,
        inputs=fake_data,
        grad_outputs=torch.ones(output_fake.shape).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0].view(batch_size, -1)

    real_grad_norm = torch.sum(real_grad ** 2, dim=1)
    fake_grad_norm = torch.sum(fake_grad ** 2, dim=1)
    w_div = torch.mean(real_grad_norm ** (p / 2) + fake_grad_norm ** (p / 2)) * k / 2

    return w_div
