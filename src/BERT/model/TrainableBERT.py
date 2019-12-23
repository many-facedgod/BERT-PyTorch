import torch.nn.functional as F

from torch import nn

from .BERT import BERT
from ..modules.Gelu import Gelu
from ..modules.BERTLayerNorm import BERTLayerNorm


class TrainableBERT(nn.Module):
    """Wrapper over BERT that allows training"""

    def __init__(self, config_dict):
        super(TrainableBERT, self).__init__()
        self.config_dict = config_dict
        self.bert = BERT(config_dict)
        bert_model_embedding_weights = self.bert.embeddings.token_embedding.weight
        self.lm_l1 = nn.Linear(config_dict["hidden_size"], config_dict["hidden_size"])
        self.lm_ln = BERTLayerNorm(config_dict)
        self.lm_l2 = nn.Linear(bert_model_embedding_weights.shape[1], bert_model_embedding_weights.shape[0])
        self.lm_l2.weight = bert_model_embedding_weights
        self.nsp_l1 = nn.Linear(config_dict["hidden_size"], 2)
        self.activation = Gelu()

    def forward(self, batch_input):
        hidden_states, cls = self.bert(batch_input)
        sentence_class = F.log_softmax(self.nsp_l1(cls), dim=-1)
        lang_model = F.log_softmax(self.lm_l2(self.lm_ln(self.activation(self.lm_l1(hidden_states)))), dim=-1)
        return {"lm_predictions": lang_model, "nsp_predictions": sentence_class}





