import torch
import math

def gelu(input):
    return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))
