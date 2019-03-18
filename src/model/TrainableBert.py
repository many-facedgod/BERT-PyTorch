from torch import nn
from Bert import Bert
from Gelu import gelu
from LayerNorm import BertLayerNorm
import torch


class TrainableBert(nn.Module):

    def __init__(self, config_dict, pretrained_bert=None):
        super(TrainableBert, self).__init__()

        if pretrained_bert is None:
            self.bert = Bert(config_dict)
        else:
            self.bert = pretrained_bert

        bert_model_embedding_weights = self.bert.embeddings.weights
        self.transform = BertPredictionTransform(config_dict)
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)

        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

        self.sentence_classification = nn.Linear(config_dict["hidden_size"], 2)

    def forward(self, input):
        shared = self.bert(input)
        sentence_class = self.sentence_classification(shared)
        lang_model = self.transform(shared)
        lang_model = self.decoder(lang_model) + self.bias
        return lang_model, sentence_class


class BertPredictionTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionTransform).__init__()
        self.linear = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.layer_norm = BertLayerNorm(config)

    def forward(self, input):
        return self.layer_norm(gelu(self.linear(input)))
