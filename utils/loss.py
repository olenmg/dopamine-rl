import torch
import torch.nn as nn


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
        target_support = y.unsqueeze(-1).tile((1, 1, 4))
        pred_support = y.unsqueeze(-1).tile((1, 1, 4)).transpose(2, 1)
        u = target_support - pred_support

        l_k = self.huber_loss(u)
        mask = (u < 0)
        q_huber_loss = (mask * l_k * (1 - quant_target)) + (~mask * l_k * quant_target)

        return torch.mean(torch.sum(q_huber_loss, dim=-1))

    def huber_loss(self, u):
        """
        Args:
            u shape: (B, n_quant, n_quant)
        """
        mask = (abs(u) <= self.kappa)
        return (mask * (u ** 2) / 2) + (~mask * self.kappa * (abs(u) - self.kappa / 2))
