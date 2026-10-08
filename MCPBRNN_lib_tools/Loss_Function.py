import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

class KGELoss(torch.nn.Module):

    def __init__(self):
        super(KGELoss, self).__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):

        sz = y_pred.shape[0]
        # Without time-lag
        x = torch.cat((y_pred, y_true), 1)
        xx = torch.transpose(x, 0, 1)
        c = torch.corrcoef(xx)       
        alpha = torch.std(y_pred)/torch.std(y_true)
        beta = torch.sum(y_pred)/torch.sum(y_true)
        cc = c[0,1].unsqueeze(0)
        a = alpha.unsqueeze(0)
        b = beta.unsqueeze(0)                
        kgeM = torch.sqrt( (cc-1)*(cc-1) + (a-1)*(a-1) + (b-1)*(b-1) )

        return kgeM