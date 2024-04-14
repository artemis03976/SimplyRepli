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

        delattr(self.module, self.param_name)
        self.module.register_parameter(self.param_name + "_origin", original_weight)
        setattr(self.module, self.param_name, original_weight.data)

    @staticmethod
    def l2_normalize(value):
        return value / (torch.norm(value) + 1e-12)

    def iterate(self, weight, power_iterations):
        u = getattr(self, 'u')
        v = getattr(self, 'v')

        for _ in range(power_iterations):
            v = self.l2_normalize(
                torch.matmul(weight.view(weight.data.shape[0], -1).data.transpose(0, 1), u)
            )
            u = self.l2_normalize(
                torch.matmul(weight.view(weight.data.shape[0], -1).data, v)
            )

        return u, v

    def spectral_norm(self):
        original_weight = getattr(self.module, self.param_name + "_origin")

        weight_h = original_weight.data.shape[0]

        u, v = self.iterate(original_weight, self.power_iterations)

        # calculate sigma
        sigma = torch.dot(u, torch.matmul(original_weight.view(weight_h, -1), v))
        # update original weight with differentiable method
        setattr(self.module, self.param_name, original_weight / sigma)

    def forward(self, *args):
        # do spectral norm on weight
        self.spectral_norm()

        return self.module.forward(*args)
