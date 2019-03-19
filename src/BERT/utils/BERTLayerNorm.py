import torch.nn as nn
import torch


class BERTLayerNorm(nn.Module):
    def __init__(self, config_dict):
        super(BERTLayerNorm, self).__init__()
        hidden_size = config_dict["hidden_size"]
        self.eps = config_dict["eps_value"]
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, inp):
        mu = inp.mean(-1, keepdim=True)
        var = ((inp - mu) ** 2).mean(-1, keepdim=True)
        return self.bias + (self.weight * ((inp - mu) / (torch.sqrt(self.eps + var))))