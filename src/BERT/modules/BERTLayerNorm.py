import torch.nn as nn


class BERTLayerNorm(nn.Module):
    def __init__(self, config_dict):
        super(BERTLayerNorm, self).__init__()
        hidden_size = config_dict["hidden_size"]
        eps = config_dict["eps_value"]
        self.layer_norm = nn.LayerNorm(hidden_size, eps=eps)

    def forward(self, inp):
        return self.layer_norm(inp)
