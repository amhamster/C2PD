# -*- coding: utf-8 -*-

from torch import nn

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.loss = nn.L1Loss(reduction='mean')

    def forward(self, out, gt):
        return self.loss(out, gt)
