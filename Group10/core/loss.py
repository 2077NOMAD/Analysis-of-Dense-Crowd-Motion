import torch.nn as nn
import torch.nn.functional as f
from torch import Tensor
import torch
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=5, reduction='mean',class_weights=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = class_weights

    def forward(self, inputs, targets):

        scaled_inputs = inputs / self.temperature
        max_logits = torch.max(scaled_inputs, dim=1, keepdim=True)[0]
        norm_logits = scaled_inputs - max_logits.detach()  # 数值稳定处理
        ce_loss = nn.functional.cross_entropy(norm_logits, targets, reduction='none')
        if self.class_weights is not None:
            weight_factor = torch.sqrt(self.class_weights.to(inputs.device)[targets])
            ce_loss = ce_loss * weight_factor * (1 + 0.5 * torch.log1p(weight_factor))
        
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def get_loss(opt):
    if opt.loss_func == 'ce':
        return nn.CrossEntropyLoss()
    elif opt.loss_func == 'pcce_ve8':
        return PCCEVE8(lambda_0=opt.lambda_0)
    else:
        raise Exception
