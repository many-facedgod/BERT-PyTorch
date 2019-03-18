from torch import nn
from ..utils.TransformerBlock import TransformerBlock
from ..utils.BERTEmbedding import BERTEmbedding


class Bert(nn.Module):
    def __init__(self, config):
        super(Bert, self).__init__()
        self.config = config
        self.embeddings = BERTEmbedding(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def forward(self, batch_input):
        input = self.embeddings(batch_input["sentences"], batch_input.get("sentence_type", None))
        all_sequences = self.encoder(input, batch_input["sentence_mask"])
        last_seq = all_sequences[-1]
        pooled_op = self.pooler(last_seq)
        return all_sequences, pooled_op


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        num_layers = config["num_layers"]
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(num_layers)])

    # 2D
    def forward(self, input, mask):
        output_saved_layers = []
        extended_attention_mask = (1.0 - mask.unsqueeze(1).unsqueeze(2)) * -1e18
        for layer in self.layers:
            input = layer(input, extended_attention_mask)
            output_saved_layers.append(input)
        return output_saved_layers


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        hidden_size = config["hidden_size"]
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, input):
        first_token = input[:, 0]
        return self.tanh(self.linear(first_token))
