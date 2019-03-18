from torch import nn

class Bert(nn.Module):
    def __init__(self,config):
        self.config = config

    def forward(self, input, mask):


