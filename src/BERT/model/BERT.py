from torch import nn
from ..modules.BERTTransformerBlock import BERTTransformerBlock
from ..modules.BERTEmbedding import BERTEmbedding
import torch.nn.functional as F


class BERT(nn.Module):
    def __init__(self, config_dict, vocab_dict, inv_vocab_dict, weights_dict=None):
        super(BERT, self).__init__()
        self.config_dict = config_dict
        self.vocab_dict = vocab_dict
        self.inv_vocab_dict = inv_vocab_dict
        assert len(self.vocab_dict) == config_dict["vocab_size"], "Mismatch in the config and the vocab"
        self.embeddings = BERTEmbedding(config_dict)
        num_layers = config_dict["num_layers"]
        self.layers = nn.ModuleList([BERTTransformerBlock(config_dict) for _ in range(num_layers)])
        self.pooler = nn.Linear(config_dict["hidden_size"], config_dict["hidden_size"])
        if weights_dict is not None:
            self.load_from_weights_dict(weights_dict)

    def load_from_weights_dict(self, weights_dict):
        assert self.config_dict["positional_learnt"], "Positional embeddings can't be sinusoid if loading"
        self.embeddings.token_embedding.weight = weights_dict["bert.embeddings.word_embeddings.weight"]
        self.embeddings.type_embedding.weight = weights_dict["bert.embeddings.token_type_embeddings.weight"]
        self.embeddings.positional_embedding.weight = weights_dict["bert.embeddings.position_embeddings.weight"]
        self.embeddings.layer_norm.layer_norm.weight = weights_dict["bert.embeddings.LayerNorm.gamma"]
        self.embeddings.layer_norm.layer_norm.bias = weights_dict["bert.embeddings.LayerNorm.beta"]
        self.pooler.weight = weights_dict["bert.pooler.dense.weight"]
        self.pooler.bias = weights_dict["bert.pooler.dense.bias"]
        for i in range(self.config_dict["num_layers"]):
            block = self.layers[i]
            block.attention.query_map.weight = weights_dict[f"bert.encoder.layer.{i}.attention.self.query.weight"]
            block.attention.query_map.bias = weights_dict[f"bert.encoder.layer.{i}.attention.self.query.bias"]
            block.attention.key_map.weight = weights_dict[f"bert.encoder.layer.{i}.attention.self.key.weight"]
            block.attention.key_map.bias = weights_dict[f"bert.encoder.layer.{i}.attention.self.key.bias"]
            block.attention.value_map.weight = weights_dict[f"bert.encoder.layer.{i}.attention.self.value.weight"]
            block.attention.value_map.bias = weights_dict[f"bert.encoder.layer.{i}.attention.self.value.bias"]
            block.attention.ff.linear.weight = weights_dict[f"bert.encoder.layer.{i}.attention.output.dense.weight"]
            block.attention.ff.linear.bias = weights_dict[f"bert.encoder.layer.{i}.attention.output.dense.bias"]
            block.attention.ff.layer_norm.weight = weights_dict[
                f"bert.encoder.layer.{i}.attention.output.LayerNorm.gamma"]
            block.attention.ff.layer_norm.bias = weights_dict[f"bert.encoder.layer.{i}.attention.output.LayerNorm.beta"]
            block.bottleneck.weight = weights_dict[f"bert.encoder.layer.{i}.intermediate.dense.weight"]
            block.bottleneck.weight = weights_dict[f"bert.encoder.layer.{i}.intermediate.dense.bias"]
            block.output.linear.weight = weights_dict[f"bert.encoder.layer.{i}.output.dense.weight"]
            block.output.linear.bias = weights_dict[f"bert.encoder.layer.{i}.output.dense.bias"]
            block.output.layer_norm.weight = weights_dict[f"bert.encoder.layer.{i}.output.LayerNorm.gamma"]
            block.output.layer_norm.bias = weights_dict[f"bert.encoder.layer.{i}.output.LayerNorm.bias"]

    def forward(self, batch_input, return_all=False):
        input = self.embeddings(batch_input["sentences"], batch_input.get("sentence_type", None))
        extended_attention_mask = (1.0 - batch_input["sentence_mask"].unsqueeze(1).unsqueeze(2)) * -1e18
        saved_output = []
        for layer in self.layers:
            input = layer(input, extended_attention_mask)
            if return_all:
                saved_output.append(input)
            else:
                saved_output = input
        cls_token = saved_output[-1][:, 0]
        pooled = F.tanh(self.pooler(cls_token))
        return saved_output, pooled
