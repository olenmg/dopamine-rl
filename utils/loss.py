import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, y):
        """
        Args:
            pred: distribution of prediction (already applied softmax, sum=1)
                shape: (B, class_num)
            y: distribution of y (already applied softmax, sum=1)
                shape: (B, class_num)
        Return:
            out, tensor(int)
        """
        return torch.mean(y * (-torch.log(pred + 1e-8)))


class QuantileHuberLoss(nn.Module):
    def __init__(self, kappa=1.0):
        super().__init__()
        self.kappa = kappa

    def forward(self, pred, y, quant_target):
        """
        Args:
            pred: predicted quantiles
                shape: (B, n_quant)
            y: target quantiles
                shape: (B, n_quant)
        Return:
            out, tensor(int)
        """
        target_quant = target_quant.unsqueeze(1)
        pred_quant = pred_quant.unsqueeze(2)
        u = target_quant - pred_quant # (B, n_quant, n_quant)

        weight = torch.abs(u.le(0.).float() - quant_target.view(1, -1, 1)) # (B, n_quant, n_quant)
        loss = weight * F.smooth_l1_loss(pred_quant, target_quant, reduction='none') # (B, n_quant, n_quant)
        loss = torch.mean(weight * loss)

        return loss
