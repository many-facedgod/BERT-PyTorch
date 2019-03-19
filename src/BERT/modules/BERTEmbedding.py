import numpy as np
import torch
import torch.nn as nn
from .BERTLayerNorm import BERTLayerNorm


class BERTEmbedding(nn.Module):

    def __init__(self, config_dict):
        super(BERTEmbedding, self).__init__()
        self.config = config_dict
        self.token_embedding = nn.Embedding(config_dict["vocab_size"], config_dict["hidden_size"], padding_idx=0)
        self.type_embedding = nn.Embedding(2, config_dict["hidden_size"])
        if config_dict["positional_learnt"]:
            self.positional_embedding = nn.Embedding(config_dict["max_sentence_length"], config_dict["hidden_size"])
        else:
            self.register_buffer("positional_embedding",
                                 BERTEmbedding._generate_sinusoid_pos_embedding(config_dict["max_sentence_length"],
                                                                                config_dict["hidden_size"]))
        self.dropout = nn.Dropout(config_dict["dropout_rate"])
        self.layer_norm = BERTLayerNorm(config_dict)

    @staticmethod
    def _generate_sinusoid_pos_embedding(max_len, embedding_dim):
        """
        Generates the temporal embeddings
        :param max_len: The length of the time-series
        :param embedding_dim: The embedding dimensions
        :return: The generated embedding matrix
        """
        embeddings = torch.zeros(max_len, embedding_dim)
        numerator = torch.arange(max_len).float().view(1, -1)
        inv_denominator = torch.exp(- torch.arange(0, embedding_dim, 2).float() * np.log(10000) / embedding_dim)
        embeddings[:, ::2] = torch.sin(numerator * inv_denominator)
        embeddings[:, 1::2] = torch.cos(numerator * inv_denominator)
        return embeddings

    def forward(self, sentences, sentence_type=None):
        length = sentences.shape[1]
        token_embedding = self.token_embedding(sentences)
        if sentence_type is None:
            sentence_embedding = self.type_embedding(torch.zeros_like(sentences))
        else:
            sentence_embedding = self.type_embedding(sentence_type)

        if self.config_dict["positional_learnt"]:
            positional_embedding = self.positional_embedding[:length].unsqueeze(0).expand_as(sentence_embedding)
        else:
            positional_embedding = self.positional_embedding(
                torch.arange(0, length, device=sentences.device)).unsqueeze(0).expand_as(sentence_embedding)
        summed_embedding = token_embedding + positional_embedding + sentence_embedding
        return self.dropout(self.layer_norm(summed_embedding))
