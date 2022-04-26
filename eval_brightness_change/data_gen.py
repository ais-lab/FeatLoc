#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 16:05:17 2022

@author: thuan
"""

from model.dataloaders import CRDataset_h5_generation
import os 
from model.superpoint import SuperPoint
from tqdm import tqdm
import torch
import numpy as np
import h5py

### 
def gen_main(change, out_path):
    # for generating the new testing data under changing condition 
    # For example: changing brightness, and shadow noise
    scene = 'chess'
    n_kpoints = 2048
    g_data_dir ="../dataset/Hierarchical_Localization/datasets/7scenes/chess"
    resize = [-1]
    superPoint_config = {
        'nms_radius': 3,
        'keypoint_threshold':0.0,
        'max_keypoints': n_kpoints,
        'pre_train': "superpoint_v1.pth",
            }
    as_half=True
    config = {"photometric":
                {"enable": True,
                "primitives": [
                    'random_brightness', 'random_contrast', 'additive_speckle_noise',
                    'additive_gaussian_noise', 'additive_shade', 'motion_blur'],
                "params":
                    {
                    "random_brightness": {"max_abs_change": 50},
                    # "additive_shade":{
                    #     "transparency_range": [0.5, 0.5],
                    #     "kernel_size_range": [20, 150]},
                    }},
            "homographic":
                {"enable": False}  # not implemented
             }
    
    config['photometric']['params']['random_brightness']['max_abs_change'] = change
    print("[INFOR]---------------- change : ", change)
    #---------------------------------END initialization----------------------------------------
    type_ = "test"

    superPoint_config['max_keypoints'] = n_kpoints
    try:
        os.mkdir(out_path)
    except:
        pass 
    print("working with scene {} - {} points".format(scene, n_kpoints))
    sp_model = SuperPoint(superPoint_config).eval()
    sp_model.cuda()
    train_poses_path = os.path.join(out_path, "ori_" + type_ + ".txt")
    data_loader = CRDataset_h5_generation(train_poses_path, g_data_dir, "cuda", resize, config)
    loader = torch.utils.data.DataLoader(data_loader)
    number_batch = len(loader)
    pbar = tqdm(loader, total=number_batch)
    with torch.no_grad():
        for img, name, size in pbar:
            name = name[0]
            pred = sp_model.forward_training({"image": img})
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            pred['image_size'] = original_size = size[0,:].numpy()
            
            if 'keypoints' in pred:
                size = np.array(img.shape[-2:][::-1])
                scales = (original_size / size).astype(np.float32)
                pred['keypoints'] = (pred['keypoints'] + .5) * scales[None] - .5
    
            if as_half:
                for k in pred:
                    dt = pred[k].dtype
                    if (dt == np.float32) and (dt != np.float16):
                        pred[k] = pred[k].astype(np.float16)
            feature_path = os.path.join(out_path, name + ".h5")
            with h5py.File(str(feature_path), 'w') as fd:
                grp = fd.create_group(name)
                for k, v in pred.items():
                    grp.create_dataset(k, data=v)
            del pred
        return out_path


