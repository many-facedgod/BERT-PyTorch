import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Feedforward import FeedForward


class BERTAttention(nn.Module):

    def __init__(self, config_dict):
        super(BERTAttention, self).__init__()
        self.n_heads = config_dict["n_heads"]
        self.head_size = config_dict["hidden_size"] // config_dict["n_heads"]
        self.hidden_size = config_dict["hidden_size"]
        self.dropout = nn.Dropout(config_dict["attention_dropout_rate"])
        self.query_map = nn.Linear(self.hidden_size, self.hidden_size)
        self.key_map = nn.Linear(self.hidden_size, self.hidden_size)
        self.value_map = nn.Linear(self.hidden_size, self.hidden_size)
        self.ff = FeedForward(config_dict)

    def forward(self, embeddings, mask):
        batch_size = len(embeddings)
        length = embeddings.shape[1]
        all_queries = self.query_map(embeddings).view(batch_size, length, self.n_heads, self.head_size).permute(0, 2, 1,
                                                                                                                3)
        all_keys = self.key_map(embeddings).view(batch_size, length, self.n_heads, self.head_size).permute(0, 2, 1, 3)
        all_values = self.value_map(embeddings).view(batch_size, length, self.n_heads, self.head_size).permute(0, 2, 1,
                                                                                                               3)
        energies = torch.matmul(all_queries, all_values.permute(0, 1, 3, 2)) / np.sqrt(self.n_heads) + mask
        weights = F.softmax(energies, dim=3)
        weights = self.dropout(weights)
        attended = torch.matmul(weights, all_keys).permute(0, 2, 1, 3).contiguous().view(batch_size, length, -1)
        return self.ff(attended)
