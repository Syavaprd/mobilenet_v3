import torch
import torch.nn as nn
import torch.nn.functional as F

class HardSigmoid(nn.Module):
    def __init__(self):
        super(HardSigmoid, self).__init__()

    def forward(self, x):
        return F.relu6(x + 3.) / 6.


class HardSwish(nn.Module):
    def __init__(self):
        super(HardSwish, self).__init__()

    def forward(self, x):
        return x * F.relu6(x + 3.) / 6.

def countParams(model):
    return sum(p.numel() for p in model.parameters())
