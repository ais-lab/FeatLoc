#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 00:21:06 2021

@author: thuan
"""

from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd 
import os 
from .utils import read_image, read_image_for_validation
import h5py
import os.path as osp


def qlog(q):
    if all(q[1:] == 0):
        q = np.zeros(3)
    else:
        q = np.arccos(q[0]) * q[1:] / np.linalg.norm(q[1:])
    return q

class CRDataset_train(Dataset):
    def __init__(self, poses_path:str, images_path:str, device='cuda', resize = [-1]):
        self.df = pd.read_csv(poses_path, header = None, sep = " ")
        self.images_path = images_path
        self.device = 'cuda'
        self.resize = resize 
        
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self, idx):
        target = self.df.iloc[idx, 1:]
        target = np.array(target).astype(float)
        img_path = self.images_path + self.df.iloc[idx,0]
        _, img, _ = read_image(img_path, self.device, self.resize,0 ,False)
        
        target = torch.Tensor(target)
        _,_,m,n = img.shape
        img = img.view(1,m,n)
        img_shape = torch.from_numpy(np.array([n,m]))
        return img, target, self.df.iloc[idx,0], img_shape


class CRDataset_h5_generation(Dataset):
    def __init__(self, poses_path:str, images_path:str, device='cuda', resize = [-1], config=dict):
        self.df = pd.read_csv(poses_path, header = None, sep = " ")
        self.images_path = images_path
        self.device = 'cuda'
        self.resize = resize 
        self.config = config
        
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_path, self.df.iloc[idx,0])
        ori_img, img, _ = read_image_for_validation(img_path, self.device, self.resize, 0 ,False, self.config)
        _,_,m,n = img.shape
        img = img.view(1,m,n)
        return img, self.df.iloc[idx,1], np.array(ori_img.shape[:2][::-1])
    


class Load_sfm_data(Dataset):
    
    def __init__(self, data_path, dtype, data_path_op = None, num = 2048 ):
        # for the true pose label.
        if not (dtype == "train" or dtype == "test"):
            raise ValueError("dtype must be (train) or (test)")
        if data_path_op != None:
            self.infor = pd.read_csv(osp.join(data_path_op, "ori_" + dtype + ".txt"), header = None, sep = " ")
        else:
            self.infor = pd.read_csv(osp.join(data_path, "ori_" + dtype + ".txt"), header = None, sep = " ")
        self.data_path = data_path
        self.num = num 
    
    def __len__(self):
        return len(self.infor)
    
    def __getitem__(self, idx):
        target_ = self.infor.iloc[idx,2:]
        target_ = np.array(target_).astype(float)
        target = np.empty(6).astype(float)
        target[:3] = target_[:3]
        q = target_[3:]
        target[3:] = qlog(q)
        target = torch.Tensor(target)
        del target_
        name = self.infor.iloc[idx,1]
        h5File = osp.join(self.data_path, name + ".h5")
        features = h5py.File(h5File, 'r')
        data = {}
        for k,v in features[name].items():    
            if k == 'image_size':
                data[k] = torch.from_numpy(v.__array__()).float()
            elif  k == 'descriptors':
                data[k] = torch.from_numpy(v.__array__()[:,:self.num]).float()
            else:
                data[k] = torch.from_numpy(v.__array__()[:self.num]).float()
            
        return data, target


class Load_sfm_data_plus(Dataset):
    
    def __init__(self, data_path):
        
        self.infor_ori = pd.read_csv(osp.join(data_path, "ori_train.txt"), header = None, sep = " ")
        self.infor_aug = pd.read_csv(osp.join(data_path, "augmented_targets.txt"), header = None, sep = " ")
        self.data_path = data_path
        self.l1 = len(self.infor_ori)
        self.l2 = len(self.infor_aug)
    def __len__(self):
        return self.l1 + self.l2 
    
    def __getitem__(self, idx):
        if idx < self.l1:
            target_ = self.infor_ori.iloc[idx,2:]
            target_ = np.array(target_).astype(float)
            target = np.empty(6).astype(float)
            target[:3] = target_[:3]
            q = target_[3:]
            target[3:] = qlog(q)
            target = torch.Tensor(target)
            del target_
            name = self.infor_ori.iloc[idx,1]
            h5File = osp.join(self.data_path, name + ".h5")
            features = h5py.File(h5File, 'r')
            data = {}
            for k,v in features[name].items():    
                data[k] = torch.from_numpy(v.__array__()).float()
        else:
            new_idx = idx - self.l1
            target_ = self.infor_aug.iloc[new_idx,1:]
            target_ = np.array(target_).astype(float)
            target = np.empty(6).astype(float)
            target[:3] = target_[:3]
            q = target_[3:]
            target[3:] = qlog(q)
            target = torch.Tensor(target)
            del target_
            name = self.infor_aug.iloc[new_idx,0]
            h5File = osp.join(self.data_path, name + "_augmented_feature" + ".h5")
            features = h5py.File(h5File, 'r')
            data = {}
            for k,v in features[name].items():    
                data[k] = torch.from_numpy(v.__array__()).float()
        return data, target
    
    
    
    
    
    
    
    
    
    
    
    
    
    