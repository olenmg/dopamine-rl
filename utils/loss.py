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
            out, shape: (B, )
        """
        return torch.mean(y * (-torch.log(pred + 1e-8)))


CUSTOM_LOSS = {
    "SoftCrossEntropyLoss": SoftCrossEntropyLoss
}
