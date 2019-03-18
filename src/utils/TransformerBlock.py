from torch import nn
from Gelu import gelu
import torch
import math
from Feedforward import FeedForward

from LayerNorm import BertLayerNorm
from Feedforward import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, config_dict):
        super(TransformerBlock).__init__()
        self.intermediate_size = config_dict["intermediate_size"]
        self.hidden_size = config_dict["hidden_size"]
        self.attention = SelfAttention(config_dict)
        self.feedforward = FeedForward(config_dict)
        self.intermediate_linear = nn.Linear(self.hidden_size, self.intermediate_size)

    def forward(self, input):
        self_att_output = self.attention(input)
        intermediate_linear_op = gelu(self.intermediate_linear(self_att_output))
        return self.feedforward(intermediate_linear_op)


class SelfAttention(nn.Module):
    def __init__(self, config_dict):
        super(SelfAttention).__init__()

    def forward(self, input):
        return self.layer_norm(input + self.dropout(self.linear(input)))
