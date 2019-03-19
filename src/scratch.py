from BERT.model.BERT import BERT
from BERT.modules.BERTLayerNorm import BERTLayerNorm
import torch.nn as nn
import torch
import numpy as np

config = {}
config["hidden_size"] = 10
config["vocab_size"] = 20
config["num_layers"] = 3
config["positional_learnt"] = True
config["n_heads"] = 2
config["attention_dropout_rate"] = 0.1
config["dropout_rate"] = 0.1
config["max_sentence_length"] = 20
config["bottleneck_size"] = 10
config["eps_value"] = 1e-12

x = np.random.randint(1, 20, size=(4, 15))
lengths = [15, 10, 11, 2]
mask = np.zeros_like(x, dtype=np.int64)
for i in range(4):
    mask[i, :lengths[i]] = 1

x = torch.LongTensor(x)
mask = torch.LongTensor(mask)
print(x)
print(mask)
batch = {}
batch["sentences"] = x
batch["sentence_mask"] = mask.float()
model = BERT(config, {}, {})
print(model(batch))