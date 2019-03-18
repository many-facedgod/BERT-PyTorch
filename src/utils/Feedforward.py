from torch import nn
from LayerNorm import BertLayerNorm


class FeedForward(nn.Module):
    def __init__(self, config_dict):
        super(FeedForward).__init__()
        hidden_size = config_dict["hidden_size"]
        dropout_rate = config_dict["dropout_rate"]
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = BertLayerNorm(config_dict)

    def forward(self, input):
        return self.layer_norm(input + self.dropout(self.linear(input)))
