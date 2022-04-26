#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 12:54:56 2022

@author: thuan
"""
from scipy.spatial.transform import Rotation as R
import random
from tqdm import tqdm
import os
import pandas as pd
import numpy as np 
import pycolmap
from Hierarchical_Localization.hloc.utils.read_write_model import (
         read_images_binary, read_model, read_images_text)
import h5py
import torch



def project_to_image(p3D, R, t, camera, eps: float = 1e-4, pad: int = 1):
    p3D = (p3D @ R.T) + (t)
    visible = p3D[:, -1] >= eps  # keep points in front of the camera
    p2D_norm = p3D[:, :-1] / p3D[:, -1:].clip(min=eps)
    ret = pycolmap.world_to_image(p2D_norm, camera._asdict())
    p2D = np.asarray(ret['image_points'])
    size = np.array([camera.width - pad - 1, camera.height - pad - 1])
    valid = np.all((p2D >= pad) & (p2D <= size), -1)
    valid &= visible
    return p2D[valid], valid
def Rt2T(R,t):
    last = np.array([0,0,0,1])
    T = np.zeros((4,4))
    T[:3,:3] = R
    T[:3,3] = t
    T[3,:] = last
    return T

def T2Rt(T):
    return T[:3,:3], T[:3,3]
def angle_X(degree):
    tmp = np.pi/180.0
    return np.array([[1,0,0],
                     [0,np.cos(tmp*degree),-np.sin(tmp*degree)],
                    [0,np.sin(tmp*degree), np.cos(tmp*degree)]])
def angle_Y(degree):
    tmp = np.pi/180.0
    return np.array([[np.cos(tmp*degree),0,np.sin(tmp*degree)],
                     [0,1,0],
                    [-np.sin(tmp*degree), 0, np.cos(tmp*degree)]])
def angle_Z(degree):
    tmp = np.pi/180.0
    return np.array([[np.cos(tmp*degree),-np.sin(tmp*degree),0],
                    [-np.sin(tmp*degree), np.cos(tmp*degree), 0],
                    [0,0,1],])
def change_T(a_x, a_y, a_z, t):
    newR = angle_X(a_x) @ angle_Y(a_y) @ angle_Z(a_z)
    return Rt2T(newR, t)

def augment(Tc1c2, img, points3D, cameras):
    R_w2c, t_w2c = img.qvec2rotmat(), img.tvec
    Twc1 = Rt2T(R_w2c,t_w2c)
    Twc2 = Twc1 @ Tc1c2
    R_w2c, t_w2c = T2Rt(Twc2)
    camera = cameras[img.camera_id]
    p3D_ids = img.point3D_ids
    visible_ids = p3D_ids != -1
    p3Ds = np.stack([points3D[i].xyz for i in p3D_ids[visible_ids]], 0)
    p2Ds, valids_projected = project_to_image(p3Ds, R_w2c, t_w2c, camera)
    pose = np.concatenate((t_w2c, np.roll(R.from_matrix(R_w2c).as_quat(),1)))
    return p2Ds, visible_ids, valids_projected, pose


def GenTc1c2(RaAngle, RaXYZ, num_gen):
    assert len(RaAngle) == 2, "The RaAngle argument must be a list of 2 elements"
    list_T = []
    list_d = [] # save the distance 
    for i in range(num_gen):
        ax = random.uniform(RaAngle[0],RaAngle[1])
        ay = random.uniform(RaAngle[0],RaAngle[1])
        az = random.uniform(RaAngle[0],RaAngle[1])
        x = random.uniform(-RaXYZ,+RaXYZ)
        y = random.uniform(-RaXYZ,+RaXYZ)
        z = random.uniform(-RaXYZ,+RaXYZ)
        list_T.append([ax,ay,az,[x,y,z]])
        distance = np.sqrt(x**2+y**2+z**2)
        list_d.append(distance)
    return list_T, list_d

def imgSFM2pandas_train(images, save_path, features, out_dir, _k, refine=False, fe_dim = 256):
    # use only for training data
    print("Generating training data")
    list_imgs = []
    all_imgs = np.zeros((1,7))
    for id_, img in tqdm(images.items()):
        tmp = np.zeros((1,7))
        tmp[0,:3] = img.tvec
        tmp[0,3:] = img.qvec
        all_imgs = np.concatenate((all_imgs, tmp))
        list_imgs.append(img.name)
    all_imgs= np.delete(all_imgs,0,0)
    m,_ = all_imgs.shape
    name_col = np.zeros((m,2)) 
    all_imgs = np.concatenate((name_col, all_imgs), axis = 1)
    pd_data = pd.DataFrame(all_imgs)
    pd_data.iloc[:,0] = list_imgs
    
    # seperate the .h5 file
    list_new_id = []
    for i in range(len(list_imgs)):
        name = list_imgs[i]
        new_name_id = "train_" + str(i)
        list_new_id.append(new_name_id)
        data= {}
        for k,v in features[name].items():    
            data[k] = v.__array__()
        if refine:
            if len(data["keypoints"]) > _k:
                data = refine_topk_points(data, _k)
            else:
                data = append_zeros(data, _k, len(data["keypoints"]), fe_dim)
        feature_path = os.path.join(out_dir, new_name_id + ".h5")
        with h5py.File(str(feature_path), 'w') as fd:
            grp = fd.create_group(new_name_id)
            for k, v in data.items():
                grp.create_dataset(k, data=v)
    pd_data.iloc[:,1] = list_new_id
    pd_data.to_csv(save_path, header =False,index =False, sep = " ")
    
def imgSFM2pandas_test(images_dir, out_dir, features_dir, img_test_list, _k, ext=".bin", refine=False, fe_dim=256):
    # use ony for testing data
    # images: consists of both train and test data 
    print("Generating testing data")
    if ext == '.bin':
        images = read_images_binary(images_dir)
    elif ext == '.txt':
        images = read_images_text(images_dir)
    else:
        raise 'ext must be .bin or .txt'
    features = h5py.File(features_dir, 'r')
    save_path = os.path.join(out_dir, "ori_test.txt")
    img_test_list = pd.read_csv(img_test_list, header =None, sep = " ")
    name2id = {image.name: i for i, image in images.items()}
    list_imgs = []
    all_imgs = np.zeros((1,7))
    for i in tqdm(range(len(img_test_list))):
        name = img_test_list.iloc[i,0]
        id_ = name2id[name]
        tmp = np.zeros((1,7))
        tmp[0,:3] = images[id_].tvec
        tmp[0,3:] = images[id_].qvec
        all_imgs = np.concatenate((all_imgs, tmp))
        list_imgs.append(images[id_].name)
    all_imgs= np.delete(all_imgs,0,0)
    m,_ = all_imgs.shape
    name_col = np.zeros((m,2)) 
    all_imgs = np.concatenate((name_col, all_imgs), axis = 1)
    pd_data = pd.DataFrame(all_imgs)
    pd_data.iloc[:,0] = list_imgs
    
    # seperate the .h5 file
    list_new_id = []
    for i in range(len(list_imgs)):
        name = list_imgs[i]
        new_name_id = "test_" + str(i)
        list_new_id.append(new_name_id)
        data= {}
        for k,v in features[name].items():    
            data[k] = v.__array__()
        if refine:
            if len(data["keypoints"]) > _k:
                data = refine_topk_points(data, _k)
            else:
                data = append_zeros(data, _k, len(data["keypoints"]), fe_dim)
        feature_path = os.path.join(out_dir, new_name_id + ".h5")
        with h5py.File(str(feature_path), 'w') as fd:
            grp = fd.create_group(new_name_id)
            for k, v in data.items():
                grp.create_dataset(k, data=v)
    pd_data.iloc[:,1] = list_new_id
    pd_data.to_csv(save_path, header =False,index =False, sep = " ")


def discard_false(valids, features):
    tmp_new_features = {}
    for k,v in features.items():
        if k == "keypoints":
            tmp_new_features[k] = v[valids]
        elif k == "descriptors":
            tmp_new_features[k] = v.T[valids].T
        elif k == "point3D_ids":
            tmp_new_features[k] = v[valids]
        elif k == "image_size" or k == "pose":
            tmp_new_features[k] = v
        elif k == "scores":
            tmp_new_features[k] = v[valids]
    return tmp_new_features


def refine_topk_points(out, _k =1024):
    scores = out["scores"]
    l = len(scores)
    scores = torch.from_numpy(scores).cuda()
    scores, indices = torch.topk(scores, _k, dim=0)
    scores = scores.detach().cpu().numpy()
    indices = indices.detach().cpu().numpy()
    valids = [False for i in range(l)]
    for i2 in range(len(indices)):
        valids[indices[i2]] = True
    return discard_false(valids, out)

def append_zeros(features, _k, c_length, fe_dim):
    addition = _k - c_length
    tmp_new_features = {}
    for k,v in features.items():
        if k == "keypoints":
            tmp_new_features[k] = np.concatenate((v, np.zeros((addition, 2))), axis =0)
        elif k == "descriptors":
            tmp_new_features[k] = np.concatenate((v.T, np.zeros((addition, fe_dim))), axis=0).T
        elif k == "image_size":
            tmp_new_features[k] = v
        else:
            tmp_new_features[k] = np.concatenate((v, np.zeros(addition)), axis =0)
    return tmp_new_features


def AugmentAll(sfm_path, features_file, outDir, RaAngle = [-30,30], RaXYZ = 0.3,\
               num_gen = 50, threshold =500, skip=10, _k =1024, refine =False, is_augment = 0, fe_dim = 256):
    

    feature_path = os.path.join(outDir, "augmented_feature.h5")
    target_path = os.path.join(outDir, "augmented_targets.txt")
    cameras, images, points3D = read_model(sfm_path)
    features = h5py.File(features_file, 'r')
    # create pandas targets file for original training data
    imgSFM2pandas_train(images, os.path.join(outDir,"ori_train.txt"), features, outDir, _k, refine = refine, fe_dim = fe_dim)
    if not is_augment:
        return 0,0
    print("Synthesizing un-seen data")
    num_points = None
    name_id = 0 
    poses = np.zeros((1,7))
    list_name_id = []
    l_valids = [] # save the number points which are valid reprojected
    l_visibles = [] # save the list of number visible points
    l_distances = []
    i_skip = 0
    for imgid, img in tqdm(images.items()):
        list_T, l_d = GenTc1c2(RaAngle, RaXYZ, num_gen)
        num_points,_ = img.xys.shape
        if i_skip > skip or i_skip == skip:
            for i in range(num_gen):
                Tc1c2 = change_T(*list_T[i])
                p2Ds, visible_ids, valids_projected, pose = augment(Tc1c2, img, points3D, cameras)
                tmp_new_features = {}
                valids_repr = len(p2Ds)
                if valids_repr < threshold:
                    continue
                addition = num_points - valids_repr
                l_valids.append(valids_repr)
                l_visibles.append(len(valids_projected))
                for k,v in features[img.name].items():
                    if k == "keypoints":
                        tmp_new_features[k] = np.concatenate((p2Ds, np.zeros((addition, 2))), axis =0)
                    elif k == "descriptors":
                        des = v.__array__().T[visible_ids][valids_projected]
                        tmp_new_features[k] = np.concatenate((des, np.zeros((addition, fe_dim))), axis=0).T
                    elif k == "image_size":
                        tmp_new_features[k] = v.__array__()
                    else:
                        scores = v.__array__()[visible_ids][valids_projected]
                        tmp_new_features[k] = np.concatenate((scores, np.zeros(addition)), axis =0)
                if len(tmp_new_features["keypoints"]) > _k:
                    tmp_new_features = refine_topk_points(tmp_new_features, _k)
                elif (len(tmp_new_features["keypoints"]) > threshold):
                    tmp_new_features = append_zeros(tmp_new_features, _k, len(tmp_new_features["keypoints"]), fe_dim)
                else:
                    continue # skip the bad augmentation
                tmp_name_id = "id_" + str(name_id)
                feature_path = os.path.join(outDir, tmp_name_id + "_augmented_feature.h5")
                with h5py.File(str(feature_path), 'w') as fd:
                    grp = fd.create_group(tmp_name_id)
                    for k, v in tmp_new_features.items():
                        grp.create_dataset(k, data=v)
                name_id = name_id + 1
                poses = np.concatenate((poses, pose.reshape((1,-1))), axis = 0)
                list_name_id.append(tmp_name_id)
                l_distances.append(l_d[i])
            i_skip = 0
        else:
            i_skip = i_skip + 1
            
    poses = np.delete(poses, 0, 0)
    tmp_l,_ = poses.shape
    name_col = np.zeros((tmp_l,1))
    poses = np.concatenate((name_col, poses), axis = 1)
    poses = pd.DataFrame(poses)
    poses.iloc[:,0] = list_name_id
    poses.to_csv(target_path ,header=False, index = False, sep = " ")
    print("Generated {} new views".format(len(l_distances)))
    print("Average distance is {} meters".format(sum(l_distances)/len(l_distances)))
    return l_valids, l_visibles