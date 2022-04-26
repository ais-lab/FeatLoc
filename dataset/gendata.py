#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 13:05:17 2022

@author: thuan
"""

from gendata_lib import imgSFM2pandas_test, AugmentAll
import os
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default="7scenes",
                    help='dataset name')
parser.add_argument("--scene", type=str, default="chess",
                    help="The number of threads employed by the data loader")
parser.add_argument("--augment", type=int, default=0,
                    help="Generate augmentation data or not")

args = parser.parse_args()


sfm_path = "Hierarchical_Localization/outputs/"+args.dataset+"/"+args.scene+"/sfm_superpoint+superglue"
features_file = "Hierarchical_Localization/outputs/"+args.dataset+"/"+args.scene+"/feats-superpoint-n4096-r1024.h5"
images_all_file_ = "Hierarchical_Localization/datasets/"+args.dataset+"/"+args.dataset+"_sfm_triangulated"
images_all_file = os.path.join(images_all_file_, args.scene, "triangulated/images.bin")
img_test_list = os.path.join(images_all_file_, args.scene, "triangulated/list_test.txt")
out_dir = os.path.join("Generated_Data/", args.scene)
try:
    os.mkdir(out_dir)
except:
    pass 
print("Working on --{}-- scene in --{}-- dataset".format(args.scene, args.dataset))
RaAngle = [-10,10]
RaXYZ = 0.5
num_gen = 1
threshold = 1200 # if number of valid projected ones is lower this threshold, then remove
skip = 0
_k = 2048

imgSFM2pandas_test(images_all_file, out_dir, features_file,img_test_list, _k)
m,n = AugmentAll(sfm_path,features_file, out_dir, RaAngle, RaXYZ, num_gen, threshold,skip, _k = _k, is_augment=args.augment)