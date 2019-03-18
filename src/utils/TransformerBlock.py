from torch import nn
import torch
import math
from LayerNorm import BertLayerNorm

class TransformerBlock(nn.Module):
    def __init__(self, config_dict):
        super(TransformerBlock).__init__()
        self.intermediate_size = config_dict["intermediate_size"]
        self.hidden_size = config_dict["hidden_size"]
        self.attention = SelfAttention(config_dict)
        self.feedforward = FeedForward(config_dict)
        self.intermediate_linear = nn.Linear(self.hidden_size, self.intermediate_size)

    def gelu(self, input):
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

    def forward(self, input):
        self_att_output = self.attention(input)
        intermediate_linear_op = self.gelu(self.intermediate_linear(self_att_output))
        return self.feedforward(intermediate_linear_op)

class SelfAttention(nn.Module):
    def __init__(self, config_dict):
        super(SelfAttention).__init__()

    def forward(self, input):
        return  self.layer_norm(input + self.dropout(self.linear(input)))
