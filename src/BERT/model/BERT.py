from torch import nn
from ..utils.BERTTransformerBlock import BERTTransformerBlock
from ..utils.BERTEmbedding import BERTEmbedding
import torch.nn.functional as F


class BERT(nn.Module):
    def __init__(self, config_dict, vocab_dict, inv_vocab_dict):
        super(BERT, self).__init__()
        self.vocab_dict = vocab_dict
        self.inv_vocab_dict = inv_vocab_dict
        assert len(self.vocab_dict) == config_dict["vocab_size"], "Mismatch in the config and the vocab"
        self.embeddings = BERTEmbedding(config_dict)
        num_layers = config_dict["num_layers"]
        self.layers = nn.ModuleList([BERTTransformerBlock(config_dict) for _ in range(num_layers)])
        self.pooler = nn.Linear(config_dict["hidden_size"], config_dict["hidden_size"])

    def load_from_file(self, path):
        pass

    def forward(self, batch_input):
        input = self.embeddings(batch_input["sentences"], batch_input.get("sentence_type", None))
        extended_attention_mask = (1.0 - batch_input["sentence_mask"].unsqueeze(1).unsqueeze(2)) * -1e18
        saved_output = []
        for layer in self.layers:
            input = layer(input, extended_attention_mask)
            saved_output.append(input)
        cls_token = saved_output[-1][:, 0]
        pooled = F.tanh(self.pooler(cls_token))
        return saved_output, pooled
