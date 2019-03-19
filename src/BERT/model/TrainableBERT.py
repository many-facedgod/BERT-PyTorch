from torch import nn
from .BERT import BERT
from ..modules.Gelu import gelu
from ..modules.BERTLayerNorm import BERTLayerNorm


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
        sentence_class = self.nsp_l1(cls)
        lang_model = self.lm_l2(self.lm_ln(gelu(self.lm_l1)))
        lang_model = self.decoder(lang_model) + self.bias
        return lang_model, sentence_class

    def load_from_weights_dict(self, weights_dict):
        self.bert.load_from_weights_dict(weights_dict)
        self.lm_l1.weight = weights_dict["cls.predictions.transform.dense.weight"]
        self.lm_l1.bias = weights_dict["cls.predictions.transform.dense.bias"]
        self.lm_ln.layer_norm.weight = weights_dict["cls.predictions.transform.LayerNorm.gamma"]
        self.lm_ln.layer_norm.bias = weights_dict["cls.predictions.transform.LayerNorm.beta"]
        self.lm_l2.weight = weights_dict["cls.predictions.decoder.weight"]
        self.lm_l2.bias = weights_dict["cls.predictions.bias"]
        self.nsp_l1.weight = weights_dict["cls.seq_relationship.weight"]
        self.nsp_l1.bias = weights_dict["cls.seq_relationship.bias"]




