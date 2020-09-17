import torch.nn as nn

class LossDSCreal(nn.Module):
    """
    Inputs: r
    """
    def __init__(self):
        super(LossDSCreal, self).__init__()
        self.relu = nn.ReLU()
        
    def forward(self, r):
        loss = self.relu(1. - r)
        return loss.mean()

class LossDSCfake(nn.Module):
    """
    Inputs: rhat
    """
    def __init__(self):
        super(LossDSCfake, self).__init__()
        self.relu = nn.ReLU()
        
    def forward(self, rhat):
        loss = self.relu(1. + rhat)
        return loss.mean()
