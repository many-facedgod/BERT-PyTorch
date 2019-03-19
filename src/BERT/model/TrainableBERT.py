from torch import nn
from .BERT import BERT
from ..modules.Gelu import gelu
from ..modules.BERTLayerNorm import BERTLayerNorm
import torch.nn.functional as F


class TrainableBERT(nn.Module):

    def __init__(self, config_dict, vocab_dict, inv_vocab_dict, weights_dict=None):
        super(TrainableBERT, self).__init__()
        self.vocab_dict = vocab_dict
        self.inv_vocab_dict = inv_vocab_dict
        self.config_dict = config_dict
        assert len(self.vocab_dict) == config_dict["vocab_size"], "Mismatch in the config and the vocab"
        self.bert = BERT(config_dict, vocab_dict, inv_vocab_dict)
        bert_model_embedding_weights = self.bert.embeddings.token_embedding.weight
        self.lm_l1 = nn.Linear(config_dict["hidden_size"], config_dict["hidden_size"])
        self.lm_ln = BERTLayerNorm(config_dict)
        self.lm_l2 = nn.Linear(bert_model_embedding_weights.shape[1], bert_model_embedding_weights.shape[0])
        self.lm_l2.weight = bert_model_embedding_weights
        self.nsp_l1 = nn.Linear(config_dict["hidden_size"], 2)
        if weights_dict is not None:
            self.load_from_weights_dict(weights_dict)

    def forward(self, batch_input):
        hidden_states, cls = self.bert(batch_input)
        sentence_class = F.log_softmax(self.nsp_l1(cls), dim=-1)
        lang_model = F.log_softmax(self.lm_l2(self.lm_ln(gelu(self.lm_l1(hidden_states)))), dim=-1)
        return {"lm_predictions": lang_model, "nsp_predictions": sentence_class}

    def load_from_weights_dict(self, weights_dict):
        self.bert.load_from_weights_dict(weights_dict)
        self.lm_l1.weight.data = weights_dict["cls.predictions.transform.dense.weight"]
        self.lm_l1.bias.data = weights_dict["cls.predictions.transform.dense.bias"]
        self.lm_ln.layer_norm.weight.data = weights_dict["cls.predictions.transform.LayerNorm.gamma"]
        self.lm_ln.layer_norm.bias.data = weights_dict["cls.predictions.transform.LayerNorm.beta"]
        self.lm_l2.weight.data = weights_dict["cls.predictions.decoder.weight"]
        self.lm_l2.bias.data = weights_dict["cls.predictions.bias"]
        self.nsp_l1.weight.data = weights_dict["cls.seq_relationship.weight"]
        self.nsp_l1.bias.data = weights_dict["cls.seq_relationship.bias"]




