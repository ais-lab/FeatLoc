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
from .optimizer import Optimizer
from .evaluator import get_errors, plot_result, qexp
import copy 
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import time 
import matplotlib.pyplot as plt


class Trainer(object):
    
    def __init__(self, model, optimizer_config, trainLoader, test_dataset,
                 criterion, configs):
        self.model = model
        self.total_params = sum(p.numel() for p in self.model.parameters())
        print("\nTotal parameters: {}".format(self.total_params))
        self.criterion = criterion[0]
        self.val_criterion = criterion[1]
        self.configs = configs
        self.n_epochs = self.configs.n_epochs
        self.optimizer = Optimizer(self.model.parameters(), **optimizer_config)
        self.logdir = os.path.join(self.configs.logdir, self.configs.scene + str(int(time.time())))
        try: 
            os.mkdir(self.logdir) 
        except OSError as error: 
            print(error)  
        self.his_loss = []
        self.his_val_loss = []
        self.his_best_val_loss = []
        self.best_epoch = 0 
        self.best_loss = 500
        self.best_meand_t = 10
        self.best_meand_q = 10

        self.pose_m, self.pose_s = 0, 1  # only for 7Scenes and 12Scenes 
        # set random seed 
        torch.manual_seed(self.configs.seed)
        if self.configs.GPUs > 0: 
            torch.cuda.manual_seed(self.configs.seed)
            self.device = 'cuda'
        
        # data loader 
        self.train_loader = torch.utils.data.DataLoader(trainLoader[0], batch_size=self.configs.batch_size, 
                                                        shuffle=self.configs.shuffle, num_workers=self.configs.num_workers)
        if configs.do_val:
            self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, 
                                                            shuffle=0, num_workers=self.configs.num_workers)
            self.L = len(test_dataset)
            print('number of test batch: ', len(self.test_loader))
        else:
            self.test_loader = None 
        print('number of train batch: ', len(self.train_loader))
        if self.configs.GPUs > 1: 
            self.model = nn.DataParallel(self.model, device_ids=range(self.configs.GPUs))
        if self.configs.GPUs > 0:
            self.model.cuda()
            self.criterion.cuda()
            self.val_criterion.cuda()

        
    def save_checkpoints(self, epoch):
        optim_state = self.optimizer.learner.state_dict() 
        checkpoint_dict = {'epoch':epoch, 'model_state_dict': self.model.state_dict(), 
                           'optim_state_dict': optim_state,
                           'criterion_state_dict': self.criterion.state_dict()}
        filename = os.path.join(self.logdir, 'epoch_{:03d}.pth.tar'.format(epoch))
        torch.save(checkpoint_dict, filename)
    
    def plot_loss(self):
        plt.plot(self.his_loss, label='train')
        plt.plot(self.his_val_loss, label="test")
        plt.plot(self.his_best_val_loss, label="best")
        plt.legend()
        plt.show()
    
    def train_sfm(self):

        number_train_batch = len(self.train_loader)
        
        start_total_time = time.time()
        total_time = 0.0
        for epoch in range(1,self.n_epochs+1):
            self.optimizer.adjust_lr(epoch) # adjust learning rate
            # SAVE
            if (epoch % self.configs.snapshot==0):
                if self.configs.save_checkpoint:
                    self.save_checkpoints(epoch)
                # self.plot_loss()
                
            # TRAIN
            self.model.train()
            train_loss = 0.0
            count = 0
            pbar = enumerate(self.train_loader)
            pbar = tqdm(pbar, total=number_train_batch)
            
            start_time = time.time() # time at begining of each epoch 
            
            for batch, (_inputs, poses_gt) in pbar:
                if self.configs.GPUs > 0:
                    for k,v in _inputs.items():
                        _inputs[k] = _inputs[k].cuda()
                    poses_gt = poses_gt.cuda()
                n_samples = poses_gt.shape[0]
                predict = self.model(_inputs)
                loss = self.criterion(predict, poses_gt)
                self.optimizer.learner.zero_grad()
                loss.backward()
                self.optimizer.learner.step()
                train_loss += loss.detach() * n_samples
                count += n_samples
                del loss
                del predict
                del poses_gt
                del _inputs

            total_batch_time = (time.time() - start_time)/60 # time at the end of each epoch 
            total_time += total_batch_time
            train_loss /= count
            self.his_loss.append(copy.deepcopy(train_loss).item())

            if self.configs.do_val and (epoch % self.configs.snapshot==0):
                self.model.eval()
                test_loss = 0.0 
                count = 0
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
                    n_samples = poses_gt.shape[0]
                    predict = self.model(_inputs)
                    loss = self.criterion(predict, poses_gt)
                            
                    predict_ar = np.concatenate((predict_ar, predict.cpu().detach().numpy()), axis = 0)
                    target_ar = np.concatenate((target_ar, poses_gt.cpu().detach().numpy()), axis = 0)
                    test_loss += loss.detach() * n_samples
                    count += n_samples
                    #
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
                        
                test_loss /= count
                predict_ar = np.delete(predict_ar, 0, 0)
                target_ar = np.delete(target_ar, 0, 0)
                self.his_val_loss.append(copy.deepcopy(test_loss).item())
                
                
                m,_ = predict_ar.shape
                name_col = np.zeros((m,1)) 
                predict_ar = np.concatenate((name_col, predict_ar), axis = 1)
                predict_ar = pd.DataFrame(predict_ar)
                _, _, meand_t, meand_q = get_errors(pred_poses, targ_poses, False)
                if (epoch % self.configs.snapshot==0):
                    if self.configs.save_checkpoint:
                        self.save_checkpoints(epoch)
                    if self.configs.scatter:
                        predict_plot = predict_ar.iloc[:,1:].to_numpy()
                        plot_result(predict_plot, target_ar, self.test_loader)
                # UPDATE best
                if self.best_loss > test_loss:
                    self.best_loss = test_loss
                    self.best_epoch = epoch
                    self.best_meand_t = meand_t
                    self.best_meand_q = meand_q
                    # self.save_checkpoints(epoch)
                if not self.best_loss == 0:    
                    if type(self.best_loss) == int:
                        self.his_best_val_loss.append(self.best_loss)
                    else:
                        self.his_best_val_loss.append(copy.deepcopy(self.best_loss).item())
                del output, target
                del predict_ar
                del name_col
                del pred_poses
                del targ_poses
            
            if epoch % self.configs.print_freq == 0:
                print("\nEpoch {} --- Loss: {} --- best_val_loss: {}\n".format(epoch, train_loss, self.best_loss))
                if self.configs.do_val:
                    if (epoch % self.configs.snapshot==0):
                        print("\n meand t {} --- meand q: {}\n".format(round(self.best_meand_t,4), round(self.best_meand_q,4)))
                    with open(os.path.join(self.logdir, 'results.txt'), "a") as myfile:
                        myfile.write(("\nEpoch {} --- Loss: {} --- best_val_loss: {}\n".format(epoch, train_loss, self.best_loss)))
                        if (epoch % self.configs.snapshot==0):
                            myfile.write("\n meand error t {} --- meand error q: {}\n".format(self.best_meand_t, self.best_meand_q))
        
        print("\nTraining Completed  --- Total training time: {} minutes\n".format(round(total_time,2)))
        print("Total time: {} minutes\n".format(round((time.time()-start_total_time)/60,2)))
        self.save_checkpoints(epoch)
        print("... Saved last epoch ...")
        
        
         
