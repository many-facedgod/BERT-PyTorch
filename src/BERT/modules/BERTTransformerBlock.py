import torch.nn as nn

from .BERTResidualFeedForward import BERTResidualFeedForward
from .BERTAttention import BERTAttention
from .Gelu import Gelu


class BERTTransformerBlock(nn.Module):
    def __init__(self, config_dict):
        super(BERTTransformerBlock, self).__init__()
        bottleneck_size = config_dict["bottleneck_size"]
        hidden_size = config_dict["hidden_size"]
        self.attention = BERTAttention(config_dict)
        self.bottleneck = nn.Linear(hidden_size, bottleneck_size)
        self.output = BERTResidualFeedForward(config_dict, True)
        self.activation = Gelu()

    def forward(self, input, mask):
        self_att_output = self.attention(input, mask)
        bottleneck_output = self.activation(self.bottleneck(self_att_output))
        return self.output(bottleneck_output, self_att_output)
