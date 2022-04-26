#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 17:56:03 2021

@author: thuan
"""

import torch 
import torch.nn as nn 
# import copy 

class PoseNetCriterion(nn.Module):
    def __init__(self, sx=-3.0, sq=-3.0, learn_smooth_term=True):
        super(PoseNetCriterion, self).__init__()
        self.sx_abs = nn.Parameter(torch.Tensor([sx]), requires_grad = bool(
            learn_smooth_term))
        self.sq_abs = nn.Parameter(torch.Tensor([sq]), requires_grad = bool(
            learn_smooth_term))
        
        #self.loss_func = nn.MSELoss()
        self.loss_func = nn.L1Loss()
    
    def forward(self, poses_pd, poses_gt):
        t = poses_pd[:,:3]
        q = poses_pd[:,3:]
        t_gt = poses_gt[:,:3]
        q_gt = poses_gt[:,3:]
        abs_t_loss = self.loss_func(t, t_gt)
        abs_q_loss = self.loss_func(q, q_gt)
        pose_loss = torch.exp(-self.sx_abs)*(abs_t_loss) + self.sx_abs \
            + torch.exp(-self.sq_abs)*(abs_q_loss) + self.sq_abs
        return pose_loss

    
    
    
    
    
    
    