#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class _Loss(nn.Module):
    def __init__(self, reduction='sum'):
        '''
        `reduction` (string, optional)
        - Specifies the reduction to apply to the output: `none` | `mean` | `sum`. 
        `none`: no reduction will be applied, 
        `mean`: the sum of the output will be divided by the number of elements in the output, 
        `sum`: the output will be summed. 

        Note: size_average and reduce are in the process of being deprecated, 
        and in the meantime,  specifying either of those two args will override reduction. 
        Default: `sum`
        '''
        super().__init__()
        assert(reduction == 'mean' or reduction ==
               'sum' or reduction == 'none')
        self.reduction = reduction

class BPRLoss(_Loss):
    def __init__(self, reduction='sum'):
        '''
        `reduction` (string, optional)
        - Specifies the reduction to apply to the output: `none` | `mean` | `sum`. `none`: no reduction will be applied, `mean`: the sum of the output will be divided by the number of elements in the output, `sum`: the output will be summed. Note: size_average and reduce are in the process of being deprecated, and in the meantime, specifying either of those two args will override reduction. Default: `sum`
        '''
        # ensure reduction in (mean，sum, none)
        super().__init__(reduction)

    def forward(self, model_output, **kwargs):
        '''
        `model_output` (tensor) - column 0 must be the scores of positive bundles/items, column 1 must be the negative.
        '''
        pred, L2_loss = model_output

        loss_1 = -torch.log(torch.sigmoid(pred.sum(-1)[:, 0] - pred.sum(-1)[:, 1]) + 1e-8)
        loss_2 = -torch.log(torch.sigmoid(pred.transpose(2, 1)[:, :, 0] - pred.transpose(2, 1)[:, :, 1]) + 1e-8).sum(1)
        loss = torch.mean(loss_1) * 1 + torch.mean(loss_2) * 1

        loss += L2_loss / kwargs['batch_size'] if 'batch_size' in kwargs else 0
        return loss

