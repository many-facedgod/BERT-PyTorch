from torch import nn
from .BERTLayerNorm import BERTLayerNorm


class BERTResidualFeedForward(nn.Module):
    def __init__(self, config_dict, is_final):
        super(BERTResidualFeedForward, self).__init__()
        self.is_final = is_final
        hidden_size = config_dict["hidden_size"]
        dropout_rate = config_dict["dropout_rate"]
        self.dropout = nn.Dropout(dropout_rate)
        if self.is_final:
            bottleneck_size = config_dict["bottleneck_size"]
            self.linear = nn.Linear(bottleneck_size, hidden_size)
        else:
            self.linear = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = BERTLayerNorm(config_dict)

    def forward(self, input, residual_link):
        return self.layer_norm(residual_link + self.dropout(self.linear(input)))
