from torch import nn
import torch
from .BERT import BERT
from ..utils.Gelu import gelu
from ..utils.BERTLayerNorm import BERTLayerNorm
import tarfile
import json


class TrainableBERT(nn.Module):

    def __init__(self, config_dict_or_path, vocab_dict, inv_vocab_dict):
        super(TrainableBERT, self).__init__()
        self.vocab_dict = vocab_dict
        self.inv_vocab_dict = inv_vocab_dict
        assert len(self.vocab_dict) == config_dict_or_path["vocab_size"], "Mismatch in the config and the vocab"
        if type(config_dict_or_path) is dict:
            self.bert = BERT(config_dict_or_path, vocab_dict, inv_vocab_dict)
            bert_model_embedding_weights = self.bert.embeddings.token_embedding.weight
            self.lm_l1 = nn.Linear(config_dict_or_path["hidden_size"], config_dict_or_path["hidden_size"])
            self.lm_ln = BERTLayerNorm(config_dict_or_path)
            self.lm_l2 = nn.Linear(bert_model_embedding_weights.shape[1], bert_model_embedding_weights.shape[0])
            self.lm_l2.weight = bert_model_embedding_weights
            self.nsp_l1 = nn.Linear(config_dict_or_path["hidden_size"], 2)
        else:
            assert type(config_dict_or_path) is str, "Need a path if not the config"
            self.bert, self.lm_l1, self.lm_ln, self.lm_l2, self.nsp_l1 = [None] * 5
            self.load_from_file(config_dict_or_path)

    def forward(self, batch_input):
        hidden_states, cls = self.bert(batch_input)
        sentence_class = self.nsp_l1(cls)
        lang_model = self.lm_l2(self.lm_ln(gelu(self.lm_l1)))
        lang_model = self.decoder(lang_model) + self.bias
        return lang_model, sentence_class

    @staticmethod
    def _make_config(loaded_config):
        config = {}
        config["hidden_size"] = loaded_config["hidden_size"]
        config["vocab_size"] = loaded_config["vocab_size"]
        config["num_layers"] = loaded_config["num_hidden_layers"]
        config["positional_learnt"] = True
        config["n_heads"] = loaded_config["num_attention_heads"]
        config["attention_dropout_rate"] = loaded_config["attention_probs_dropout_prob"]
        config["dropout_rate"] = loaded_config["hidden_dropout_prob"]
        config["max_sentence_length"] = loaded_config["max_position_embeddings"]
        config["bottleneck_size"] = loaded_config["intermediate_size"]
        config["eps_value"] = 1e-12
        return config

    def load_from_file(self, bert_path):
        tf = tarfile.open(bert_path)
        config_file = tf.getmember("./bert_config.json")
        config = TrainableBERT._make_config(json.loads(config_file.read()))
        assert config["vocab_size"] == len(self.vocab_dict), "Loaded config and vocab don't match"
        weights_file = None
        for member in tf.getmembers():
            if member != config_file:
                weights_file = member
        assert weights_file is not None, "No weights file available"
        weights_dict = torch.load(weights_file)
        self.bert = BERT(config, self.vocab_dict, self.inv_vocab_dict)
        self.lm_l1 = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.lm_ln = BERTLayerNorm(config)
        self.lm_l2 = nn.Linear(config["hidden_size"], config["vocab_size"])
        self.nsp_l1 = nn.Linear(config["hidden_size"], 2)
        self.lm_l1.weight = weights_dict["cls.predictions.transform.dense.weight"]
        self.lm_l1.bias = weights_dict["cls.predictions.transform.dense.bias"]
        self.lm_ln.weight = weights_dict["cls.predictions.transform.LayerNorm.gamma"]
        self.lm_ln.bias = weights_dict["cls.predictions.transform.LayerNorm.beta"]
        self.lm_l2.weight = weights_dict["cls.predictions.decoder.weight"]
        self.lm_l2.bias = weights_dict["cls.predictions.bias"]
        self.nsp_l1.weight = weights_dict["cls.seq_relationship.weight"]
        self.nsp_l1.bias = weights_dict["cls.seq_relationship.bias"]


