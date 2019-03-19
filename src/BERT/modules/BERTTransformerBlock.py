
from .BERTResidualFeedForward import BERTResidualFeedForward
from .BERTAttention import BERTAttention
import torch.nn as nn
from .Gelu import gelu


class BERTTransformerBlock(nn.Module):
    def __init__(self, config_dict):
        super(BERTTransformerBlock, self).__init__()
        bottleneck_size = config_dict["bottleneck_size"]
        hidden_size = config_dict["hidden_size"]
        self.attention = BERTAttention(config_dict)
        self.bottleneck = nn.Linear(hidden_size, bottleneck_size)
        self.output = BERTResidualFeedForward(config_dict, True)

    def forward(self, input):
        self_att_output = self.attention(input)
        bottleneck_output = gelu(self.bottleneck(self_att_output))
        return self.output(bottleneck_output, self_att_output)
