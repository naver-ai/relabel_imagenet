import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftTargetCrossEntropy(nn.Module):
    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()
