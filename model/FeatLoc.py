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

class KeypointEncoder(nn.Module):
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, keypoints, scores):
        inputs = [keypoints.transpose(1,2), scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim = 1))

class MainModel(nn.Module):

    default_config = {
        'descriptor_dim': 256,
        'keypoint_encoder': [2, 32, 64, 128, 256],
        'num_hidden':2048,
        'num_hiden_2':40,
    }

    def __init__(self, config={}):
        super().__init__()
        self.config = {**self.default_config,**config}
        self.keypoints_encoder = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])
        self.conv1 = nn.Conv1d(256, self.config['num_hidden'], 1)
        
        self.fc1 = nn.Linear(self.config['num_hidden'], self.config['num_hiden_2'])
        self.fc3_r = nn.Linear(self.config['num_hiden_2'], 3)
        self.fc3_t = nn.Linear(self.config['num_hiden_2'], 3)

    def forward(self, data):
        descpt = data['descriptors']
        keypts = data['keypoints']
        scores = data['scores']
        # normalize keypoints 
        keypts = normalize_keypoints(keypts, data['image_size'][0,:])
        # Keypoint MLP encoder
        key_encodes = self.keypoints_encoder(keypts, scores)
        descpt = descpt + key_encodes

        out = F.relu(self.conv1(descpt))
        out = nn.MaxPool1d(out.size(-1))(out)
        out = nn.Flatten(1)(out)
        out = F.relu(self.fc1(out))
        out_r = self.fc3_r(out)
        out_t = self.fc3_t(out)
        
        return torch.cat([out_t, out_r], dim = 1)


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        