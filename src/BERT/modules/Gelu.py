import math

import torch
import torch.nn as nn


class Gelu(nn.Module):
    """GELU activation by Hendrycks et. al.: https://arxiv.org/abs/1606.08415"""

    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
