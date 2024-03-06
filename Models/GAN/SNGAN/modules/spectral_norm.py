import torch
import torch.nn as nn


class SpectralNorm(nn.Module):
    def __init__(self, module, param_name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()

        self.module = module
        self.param_name = param_name
        self.power_iterations = power_iterations

        self.init_params()

    def init_params(self):
        # get weight in the layer
        original_weight = getattr(self.module, self.param_name)

        weight_h = original_weight.data.shape[0]
        weight_w = original_weight.view(weight_h, -1).data.shape[1]

        # init u and v
        self.register_buffer('u', original_weight.data.new(weight_h).normal_(0, 1))
        self.register_buffer('v', original_weight.data.new(weight_w).normal_(0, 1))

    def l2_normalize(self, value):
        return value / (torch.norm(value) + 1e-12)

    def spectral_norm(self):
        original_weight = getattr(self.module, self.param_name)
        weight_h = original_weight.data.shape[0]

        for _ in range(self.power_iterations):
            # update u and v
            self.v.data = self.l2_normalize(
                torch.matmul(torch.t(original_weight.view(weight_h, -1).data), self.u.data)
            )
            self.u.data = self.l2_normalize(
                torch.matmul(original_weight.view(weight_h, -1).data, self.v.data)
            )

        # calculate sigma
        sigma = torch.dot(self.u.data, torch.matmul(original_weight.view(weight_h, -1).data, self.v.data))
        # update original weight (do spectral norm)
        original_weight.data = original_weight.data / sigma

    def forward(self, *args):
        # do spectral norm on weight
        self.spectral_norm()

        return self.module.forward(*args)
