#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 18:40:01 2021

@author: thuan
"""
import sys
sys.path.insert(0, '../')
import torch
import torch.nn as nn 
import copy 
import os
from .utils import quaternion_angular_error
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_result(pred_poses, targ_poses, data_set):
    # this function is original from https://github.com/NVlabs/geomapnet
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    # plot on the figure object
    ss = max(1, int(len(data_set) / 1000))  # 100 for stairs
    # scatter the points and draw connecting line
    x = np.vstack((pred_poses[::ss, 0].T, targ_poses[::ss, 0].T))
    y = np.vstack((pred_poses[::ss, 1].T, targ_poses[::ss, 1].T))
    z = np.vstack((pred_poses[::ss, 2].T, targ_poses[::ss, 2].T))
    for xx, yy, zz in zip(x.T, y.T, z.T):
      ax.plot(xx, yy, zs=zz, c='gray', alpha=0.6)
    ax.scatter(x[0, :], y[0, :], zs=z[0, :], c='r', depthshade=0, alpha=0.8)
    ax.scatter(x[1, :], y[1, :], zs=z[1, :], c='g', depthshade=0, alpha=0.8)
    ax.view_init(azim=119, elev=13)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    plt.show()

def get_errors(target, predict, show = True):
    target_t = target[:,:3]
    target_q = target[:,3:]
    predict_t = predict[:,:3]
    predict_q = predict[:,3:]
    t_criterion = lambda t_pred, t_gt: np.linalg.norm(t_pred - t_gt)
    q_criterion = quaternion_angular_error
    t_loss = np.asarray([t_criterion(p, t) for p, t in zip(predict_t,
                                                       target_t)])
    q_loss = np.asarray([q_criterion(p, t) for p, t in zip(predict_q,
                                                           target_q)])
    if show:
        print ('Error in translation: median {:3.2f} m,  mean {:3.2f} m\n' \
            'Error in rotation: median {:3.2f} degrees, mean {:3.2f} degree'.format(np.median(t_loss), np.mean(t_loss),
                            np.median(q_loss), np.mean(q_loss)))
    return t_loss, q_loss, np.median(t_loss), np.median(q_loss)

def qexp(q):
    n = np.linalg.norm(q)
    q = np.hstack((np.cos(n), np.sinc(n/np.pi)*q))
    return q

class Evaluator(object):
    
    def __init__(self, model, test_dataset, configs):
        self.checkpoint_file = configs.checkpoint
        self.model = model
        self.total_params = sum(p.numel() for p in self.model.parameters())
        print("\nTotal parameters: {}".format(self.total_params))
        self.configs = configs

        self.pose_m, self.pose_s = 0, 1  # only for 7Scenes and 12Scenes 
        # set random seed 
        torch.manual_seed(self.configs.seed)
        if self.configs.GPUs > 0:
            torch.cuda.manual_seed(self.configs.seed)
            self.device = 'cuda'

        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, 
                                                        shuffle=0, num_workers=self.configs.num_workers)
        self.L = len(test_dataset)
        print('number of test batch: ', len(self.test_loader))
        self.model = nn.DataParallel(self.model, device_ids=range(self.configs.GPUs))
        self.model.cuda()
        self.load_model()

    def adapt_load_state_dict(self, state_dict):
        new_state_dict = copy.deepcopy(self.model.state_dict())
        shape_conflicts = []
        missed = []
    
        for k, v in new_state_dict.items():
            if k in state_dict:
                if v.size() == state_dict[k].size():
                    new_state_dict[k] = state_dict[k]
                else:
                    shape_conflicts.append(k)
            else:
                missed.append(k)
    
        if(len(missed) > 0):
            print("Warning: The flowing parameters are missed in checkpoint: ")
            print(missed)
        if (len(shape_conflicts) > 0):
            print(
                "Warning: The flowing parameters are fail to be initialized due to the shape conflicts: ")
            print(shape_conflicts)
    
        self.model.load_state_dict(new_state_dict)
        
    def load_model(self):
        if os.path.isfile(self.checkpoint_file):
            loc_func = None if self.configs.GPUs > 0 else lambda storage, loc: storage
            print("load ", self.checkpoint_file)
            checkpoint = torch.load(self.checkpoint_file, map_location=loc_func)
            # load model 
            self.adapt_load_state_dict(checkpoint.get('model_state_dict', checkpoint))

        else:
            raise "Error: Can not load the model, because the path is not existed"
    
    def eval_sfm(self):
        self.model.eval()
        pred_poses = np.zeros((self.L, 7))  # store all predicted poses
        targ_poses = np.zeros((self.L, 7))  # store all target poses
        pbar = enumerate(self.test_loader)
        number_test_batch = len(self.test_loader)
        pbar = tqdm(pbar, total=number_test_batch)
        predict_ar = np.zeros((1,6))
        target_ar = np.zeros((1,6))

        for batch, (_inputs, poses_gt) in pbar:
            if self.configs.GPUs > 0:
                for k,v in _inputs.items():
                    _inputs[k] = _inputs[k].cuda(non_blocking=True)
                poses_gt = poses_gt.cuda(non_blocking=True)
            predict = self.model(_inputs)

            predict_ar = np.concatenate((predict_ar, predict.cpu().detach().numpy()), axis = 0)
            target_ar = np.concatenate((target_ar, poses_gt.cpu().detach().numpy()), axis = 0)

            s = predict.size()
            output = predict.cpu().data.numpy().reshape((-1, s[-1]))
            target = poses_gt.cpu().data.numpy().reshape((-1, s[-1]))
            
            # normalize the predicted quaternions
            q = [qexp(p[3:]) for p in output]
            output = np.hstack((output[:, :3], np.asarray(q)))
            q = [qexp(p[3:]) for p in target]
            target = np.hstack((target[:, :3], np.asarray(q)))
            # un-normalize the predicted and target translations
            output[:, :3] = (output[:, :3] * self.pose_s) + self.pose_m
            target[:, :3] = (target[:, :3] * self.pose_s) + self.pose_m
            # take the middle prediction
            pred_poses[batch, :] = output[int(len(output) / 2)]
            targ_poses[batch, :] = target[int(len(target) / 2)]

        predict_ar = np.delete(predict_ar, 0, 0)
        target_ar = np.delete(target_ar, 0, 0)
        
        m,_ = predict_ar.shape
        name_col = np.zeros((m,1)) 
        predict_ar = np.concatenate((name_col, predict_ar), axis = 1)
        predict_ar = pd.DataFrame(predict_ar)
        t_loss, q_loss, meand_t, meand_q = get_errors(pred_poses, targ_poses, False)
        predict_plot = predict_ar.iloc[:,1:].to_numpy()
        print("\nMedian error in translation = {} m".format(round(meand_t,4)))
        print("Median error in rotation    = {} degrees".format(round(meand_q,4)))
        if self.configs.is_plot:
            plot_result(predict_plot, target_ar, self.test_loader)
        return meand_t, meand_q
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

    