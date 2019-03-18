from torch import nn
import torch
import math

class TransformerBlock(nn.Module):
    def __init__(self):
        super(TransformerBlock).__init__()

    def gelu(self, input):
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

    def forward(self):


class FeedForward(nn.Module):
    def __init__(self, config_dict):
        super(FeedForward).__init__()
        hidden_size = config_dict["hidden_size"]
        dropout_rate = config_dict["dropout_rate"]
        self.dropout = nn.Dropout(dropout_rate)

        self.linear = nn.Linear(hidden_size, hidden_size)



class SelfAttention(nn.Module):
    def __init__(self, config_dict):
        super(SelfAttention).__init__()

    def forward(self, input):


class BertLayerNorm(nn.Module):
    def __init__(self, config_dict):
        super(BertLayerNorm).__init__()
        hidden_size = config_dict["hidden_size"]
        self.eps = config_dict["eps_value"]
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, inp):
        mu = inp.mean(-1,keepdim = True)
        inter = (inp - mu).pow(2).mean(-1,keepdim = True)
        return self.bias + (self.weight *(inp - mu)/(torch.sqrt(self.eps + inter)))

