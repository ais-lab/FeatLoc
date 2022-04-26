#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 23:37:44 2021

@author: thuan
"""


from model.trainer import Trainer
from model.criterion import PoseNetCriterion
from model.dataloaders import Load_sfm_data, Load_sfm_data_plus
import os
import argparse



parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0,
                    help='')
parser.add_argument('--GPUs', type=int, default=1,
                    help='The number of GPUs employed.')

parser.add_argument("--augment",type=int,default=1,choices=[0,1],
                    help="learn with data augmentation or not")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--shuffle", type=int, choices=[0, 1], default=1)
parser.add_argument("--num_workers", type=int, default=8,
                    help="The number of threads employed by the data loader")
# optimizer
parser.add_argument("--sx", type=float, default=-3.0,
                    help="Smooth term for translation")
parser.add_argument("--sq", type=float, default=-3.0,
                    help="Smooth term for rotation")
parser.add_argument("--learn_sxsq", type=int,
                    choices=[0, 1], default=1, help="whether learn sx, sq")
parser.add_argument("--optimizer", type=str,
                    choices=['sgd', 'adam', 'rmsprop'], default='adam', help="The optimization strategy")
parser.add_argument("--lr", type=float, default=3e-4,
                    help="Base learning rate.")
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--lr_decay", type=float, default=1,    
                    help="The decaying rate of learning rate")
parser.add_argument('--n_epochs', type=int, default=200,
                    help='The # training epochs')
# evaluate
parser.add_argument('--do_val', type=int,
                    choices=[0, 1], default=1, help='Whether do validation when training')
parser.add_argument('--scatter', type=int,
                    choices=[0, 1], default=0, help='Whether scatter the testing data, only for using spyder to train')
parser.add_argument('--snapshot', type=int, default=5,
                    help='The snapshot frequency')
# log
parser.add_argument('--logdir', type=str, default='results/',
                    help='The directory of logs')
parser.add_argument('--print_freq', type=int, default=1,
                    help='Print frequency every n epoch')
parser.add_argument('--save_checkpoint', type=int, default = 1,
                    help = 'save the model at snapshot')

# dataloader
parser.add_argument('--scene',type=str,default="apt1_living",
                    help="name of 7Scences Dataset")
parser.add_argument('--version', type=int, default=2, choices=[0,1,2],
                    help='The version will be trained, 0-FeatLoc, 1-FeatLoc+, 2-FeatLoc++')


args = parser.parse_args()


# dataset 
datadir = os.path.join("dataset/Generated_Data", args.scene)
if args.augment:
    train_loader = Load_sfm_data_plus(datadir)
else:    
    train_loader = Load_sfm_data(datadir, "train")
    
if args.do_val:
    test_loader = Load_sfm_data(datadir, "test")
else:
    test_loader = None 

# model 
if args.version == 0:
    import model.FeatLoc as v0
    print("model version: FeatLoc")
    model = v0.MainModel()
    criterion = PoseNetCriterion(args.sx, args.sq, args.learn_sxsq)
    val_criterion = criterion

elif args.version == 1:
    import model.FeatLocP as v1
    print("model version: FeatLoc+")
    model = v1.MainModel()
    criterion = PoseNetCriterion(args.sx, args.sq, args.learn_sxsq)
    val_criterion = criterion 
    
elif args.version == 2:
    import model.FeatLocPP as v2
    print("model version: FeatLoc++")
    model = v2.MainModel()
    criterion = PoseNetCriterion(args.sx, args.sq, args.learn_sxsq)
    val_criterion = criterion
else:
    raise "Doesn't exist this model"

optimizer_configs = {
    'method': args.optimizer,
    'base_lr': args.lr,
    'weight_decay': args.weight_decay,
    'lr_decay': args.lr_decay,
    'lr_stepvalues': [k/4*args.n_epochs for k in range(1, 5)]
}
# train
trainLoader = [train_loader]
cri = [criterion, val_criterion]
trainer = Trainer(model, optimizer_configs, trainLoader , test_loader, cri, args)
trainer.train_sfm()




























