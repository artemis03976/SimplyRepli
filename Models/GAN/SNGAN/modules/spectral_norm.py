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
        u = original_weight.data.new(weight_h).normal_(0, 1)
        v = original_weight.data.new(weight_w).normal_(0, 1)
        self.register_buffer('u', self.l2_normalize(u))
        self.register_buffer('v', self.l2_normalize(v))
        # pre-iterate to fit better
        # self.iterate(original_weight, power_iterations=15)

    @staticmethod
    def l2_normalize(value):
        return value / (torch.norm(value) + 1e-12)

    def iterate(self, weight, power_iterations):
        for _ in range(power_iterations):
            self.v.data = self.l2_normalize(
                torch.matmul(weight.view(weight.data.shape[0], -1).data.transpose(0, 1), self.u.data)
            )
            self.u.data = self.l2_normalize(
                torch.matmul(weight.view(weight.data.shape[0], -1).data, self.v.data)
            )

    @torch.no_grad()
    def spectral_norm(self):
        original_weight = getattr(self.module, self.param_name)
        weight_h = original_weight.data.shape[0]

        self.iterate(original_weight, self.power_iterations)

        # calculate sigma
        sigma = torch.dot(self.u, torch.matmul(original_weight.view(weight_h, -1), self.v))
        # update original weight with non-differentiable method (do spectral norm)
        original_weight.data = original_weight.data / sigma.expand_as(original_weight)

    def forward(self, *args):
        # do spectral norm on weight
        self.spectral_norm()

        return self.module.forward(*args)
