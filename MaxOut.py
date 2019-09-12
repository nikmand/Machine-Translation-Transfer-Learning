import torch
from torch import nn


# class Maxout(nn.Module):
#     def __init__(self, pool_size=2):
#         super().__init__()
#         self._pool_size = pool_size
#
#     def forward(self, x):
#         assert x.shape[1] % self._pool_size == 0, \
#             'Wrong input last dim size ({}) for Maxout({})'.format(x.shape[1], self._pool_size)
#         m, i = x.view(*x.shape[:1], x.shape[1] // self._pool_size, self._pool_size, *x.shape[2:]).max(2)
#         return m


class Maxout(nn.Module):
    def __init__(self, layer_input_dim, layer_output_dim, pool_size=2):
        super().__init__()
        self.input_dim = layer_input_dim
        self.output_dim = layer_output_dim
        self.pool_size = pool_size
        self.fc = nn.Linear(self.input_dim, self.output_dim * self.pool_size)

    def forward(self, x):
        assert x.shape[1] % self.pool_size == 0, \
            'Wrong input last dim size ({}) for Maxout({})'.format(x.shape[1], self._pool_size)
        affine_output = self.fc(x)
        # Compute and apply the proper shape for the max.
        shape = list(x.size())
        shape[-1] = self.output_dim
        shape.append(self.pool_size)

        maxed_output = torch.max(affine_output.view(*shape), dim=-1)[0]
        return maxed_output
