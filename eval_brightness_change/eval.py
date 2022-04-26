"""
Created on Sat Jun 12 23:37:44 2021

This evaluation file for testing robust of localization results when changing the 


@author: thuan
"""
import sys
sys.path.insert(0, '../')
from model.evaluator import Evaluator
from model.dataloaders import Load_sfm_data
import argparse
import pandas as pd 
import model.FeatLocPP as v2
from data_gen import gen_main    
import numpy as np 


parser = argparse.ArgumentParser()


parser.add_argument("--num_workers", type=int, default=8,
                    help="The number of threads employed by the data loader")
parser.add_argument('--seed', type=int, default=0,
                    help='')
parser.add_argument('--GPUs', type=int, default=2,
                    help='The number of GPUs employed.')
parser.add_argument('--checkpoint', type=str, default="../results/7scenes_chess_featloc++au.pth.tar",
                    help='The checkpoint file')
# log
#_-----------------------
parser.add_argument('--datadir_op',type=str,default='../dataset/Generated_Data/chess')
parser.add_argument('--datadir',type=str,default='condition_test/chess')
parser.add_argument('--is_plot', type=int, default=0,
                    help="visualization or not")


args = parser.parse_args()
list_changes = np.arange(0,121,30)
list_meant = []
list_meanq = []

for change in list_changes:
    datadir_ = gen_main(change, args.datadir)
    print(datadir_)
    print("[INFOR]---------------- change : ", change)
    test_loader = Load_sfm_data(args.datadir, "test", args.datadir_op)
    model = v2.MainModel()
    evaler = Evaluator(model , test_loader, args)
    m,n = evaler.eval_sfm()
    list_meant.append(m)
    list_meanq.append(n)
    del model

list_meant = pd.DataFrame(list_meant)
list_meanq = pd.DataFrame(list_meanq)
list_changes = pd.DataFrame(list_changes)
list_meant.to_csv("list_meant.txt", header=False, index = False, sep = " ")
list_meanq.to_csv("list_meanq.txt", header=False, index = False, sep = " ")
list_changes.to_csv("list_changes.txt", header=False, index = False, sep = " ")

import matplotlib.pyplot as plt
plt.figure()
plt.plot(list(list_changes.iloc[:,0]),list(list_meant.iloc[:,0]), label='Translation error .vs Degree of Brightneses')
plt.figure()
plt.plot(list(list_changes.iloc[:,0]), list(list_meanq.iloc[:,0]), label='Rotation error .vs Degree of Brightneses')
print(list_meant)
print(list_meanq)
plt.show()



























