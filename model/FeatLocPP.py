#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 14:44:54 2021
keypoints encoder are develop based on SuperGlue work https://arxiv.org/abs/1911.11763
@author: thuan 
"""


import torch 
from torch import nn
import torch.nn.functional as F
from pointnet2_ops.pointnet2_modules import PointnetSAModuleMSG

BN_MOMENTUM = 0.1

def MLP(channels: list, do_bn=False):
    # Multi layer perceptron 
    n = len(channels)
    layers = []
    for i in range(1,n):
        layers.append(
            nn.Conv1d(channels[i-1], channels[i], kernel_size = 1, bias =True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i], momentum=BN_MOMENTUM))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def normalize_keypoints(kpoints, image_shape):
    # Normalize the keypoints locations based on the image shape
    width, height = image_shape
    one = kpoints.new_tensor(1) 
    size = torch.stack([one*width, one*height])[None]
    center = size/2
    scaling = size.max(1, keepdim = True).values*0.7 # multiply with 0.7 because of discarded area when extracting the feature points
    return (kpoints- center[:,None,:]) / scaling[:,None,:]

    

class MainModel(nn.Module):

    default_config = {
        'descriptor_dim': 256,
        'keypoint_encoder': [32, 64, 128, 256],
        'num_hidden':2048,
        'num_hiden_2':40,
    }

    def __init__(self, config={}):
        super().__init__()
        self.config = {**self.default_config,**config}
        self.conv1 = nn.Conv1d(512+128, self.config['num_hidden'], 1)
        self.fc3_r = nn.Linear(40, 3)
        self.fc3_t = nn.Linear(40, 3)
        
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=1024,
                radii=[0.025, 0.05, 0.1],
                nsamples=[32, 16, 8],
                mlps=[[256, 64, 64, 128], [256, 128, 128, 256], [256, 128, 128, 256]],
                use_xyz=True,
                bn=False
            )
            )
        
        self.fc_layer = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256, bias=False),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 40),
            nn.LeakyReLU(0.2),
        )

    def forward(self, data):
        descpt = data['descriptors']
        keypts = data['keypoints']

        # normalize keypoints 
        keypts = normalize_keypoints(keypts, data['image_size'][0,:])
        
        m,n,_ = keypts.shape
        zeros = torch.zeros(m,n,1).cuda(non_blocking=True)
        keypts = torch.cat((keypts, zeros), 2)

        for i in range(len(self.SA_modules)):
            keypts, descpt = self.SA_modules[i](keypts, descpt)

        out = F.relu(self.conv1(descpt))
        out = nn.MaxPool1d(descpt.size(-1))(out)
        out = nn.Flatten(1)(out)
        out = self.fc_layer(out)
        out_r = self.fc3_r(out)
        out_t = self.fc3_t(out)
        
        return torch.cat([out_t, out_r], dim = 1)
