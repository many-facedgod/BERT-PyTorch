import torch

from torch import nn

from ..modules.BERTTransformerBlock import BERTTransformerBlock
from ..modules.BERTEmbedding import BERTEmbedding


class BERT(nn.Module):
    """BERT by Devlin et. al: https://arxiv.org/abs/1810.04805"""
    def __init__(self, config_dict):
        super(BERT, self).__init__()
        self.config_dict = config_dict
        self.embeddings = BERTEmbedding(config_dict)
        num_layers = config_dict["num_layers"]
        self.layers = nn.ModuleList([BERTTransformerBlock(config_dict) for _ in range(num_layers)])
        self.pooler = nn.Linear(config_dict["hidden_size"], config_dict["hidden_size"])

    def forward(self, batch_input, return_all=False):
        input_ = self.embeddings(batch_input["sentences"], batch_input.get("sentence_type", None))
        extended_attention_mask = (1.0 - batch_input["sentence_mask"].unsqueeze(1).unsqueeze(2)) * -1e18
        saved_output = []
        for layer in self.layers:
            input_ = layer(input_, extended_attention_mask)
            if return_all:
                saved_output.append(input_)
            else:
                saved_output = input_
        if return_all:
            cls_token = saved_output[-1][:, 0]
        else:
            cls_token = saved_output[:, 0]
        pooled = torch.tanh(self.pooler(cls_token))
        return saved_output, pooled
