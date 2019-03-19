import numpy as np
import torch.nn as nn


class BERTLoss(nn.Module):

    def __init__(self, lm_weight=1.0, nsp_weight=1.0):
        """
        :param lm_weight: The weight for the LM loss
        :param nsp_weight: The weight for the NSP loss
        """
        super(BERTLoss, self).__init__()
        self.lm_weight = lm_weight
        self.nsp_weight = nsp_weight

    def forward(self, predictions, labels):
        lm_labels = labels["lm_labels"]
        nsp_labels = labels["nsp_labels"]
        lm_predictions = predictions["lm_predictions"]
        nsp_predictions = predictions["nsp_predictions"]
        batch_size = len(nsp_labels)
        n_masks = len(lm_labels[0])
        lm_loss = -lm_predictions[lm_labels].sum() / n_masks
        nsp_loss = -nsp_predictions[np.arange(batch_size), nsp_labels].sum() / batch_size
        return self.lm_weight * lm_loss + self.nsp_weight * nsp_loss
