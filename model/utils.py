#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 23:37:44 2021

@author: thuan
"""
import cv2
import torch
import pandas as pd
import numpy as np


def scatter_trajectory(data, out_link):
    # sort the data into correct order 
    # for the purpose of trajectory plot
    # ex: 
    '''
        22.jpg 1 2 3
        1.jpg  2 1 1 
        ... 
        ==> 
        1.jpg  2 1 1
        22.jpg 1 2 3 
        ... 
    '''
    
    df = pd.DataFrame(index=range(365),columns=range(7))
    t = 0;
    for i in range(371):
        for ii in range(365):
            if i == int(data.iloc[ii,0].replace(".jpg","")):
                df.iloc[t,0:7] =  data.iloc[ii,1:8]
                t = t + 1
    print(df)
    df.to_csv(out_link, header = False, sep = " ", index = False)
    
    '''plt.scatter(data.iloc[:,1], data.iloc[:,2])
    plt.title('Scatter plot pythonspot.com')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()'''
    
    
def getRealCor(gTruth="", vSfM=""):
    # get the real world coordinate system of TUM dataset 
    # which will be used to convert visualSfM to world system using GCP function 
    # Input:
    #       gTruth: the groundtruth file of TUM dataset
    #       vSfM: is the saving abriviation of TUMdataset
    # output: 
    #   a .gcp file such that each line gives: filename X Y Z
    vSfM = "/home/thuan/Desktop/visual_slam/Data_for_superglue/TUM_images.txt"
    gTruth =  "/home/thuan/Downloads/rgbd_dataset_freiburg2_desk/groundtruth.txt"
    save_file = "/home/thuan/Desktop/TUM_images_SIFT/gcp3.gcp"
    g = pd.read_csv(gTruth, sep = " ")
    length_g, _ = g.shape 
    v = pd.read_csv(vSfM, sep = ",")
    step = 2
    nImg = 20
    df = pd.DataFrame(index=range(nImg),columns=range(4)) # filename X Y Z
    i = 0 
    ii = 0
    while (i < nImg):
        tmp_img = v.iloc[ii,1] # ex: 1.jpg
        tmp_time = v.iloc[ii,0].replace(".jpg","")
        tmp_time = float(tmp_time)
        for tmp_i in range(length_g):
            if tmp_i == 0:
                if (tmp_time < g.iloc[tmp_i,0]):
                    df.iloc[i,0] = tmp_img
                    df.iloc[i,1:4] = g.iloc[tmp_i,1:4]
                    break
            else:
                if (tmp_time > g.iloc[tmp_i-1,0]) and (tmp_time < g.iloc[tmp_i,0]):
                    df.iloc[i,0] = tmp_img
                    if (tmp_time - g.iloc[tmp_i-1,0]) > (g.iloc[tmp_i,0] - tmp_time):
                        df.iloc[i,1:4] = g.iloc[tmp_i,1:4]
                    else:
                        df.iloc[i,1:4] = g.iloc[tmp_i-1,1:4]
                    break
        i = i + 1
        ii = ii + step
    df.to_csv(save_file, header=False, sep = " ", index = False)



def process_resize(w, h, resize):
    assert(len(resize) > 0 and len(resize) <= 2)
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]

    # Issue warning if resolution is too small or too large.
    if max(w_new, h_new) < 160:
        print('Warning: input resolution is very small, results may vary')
    elif max(w_new, h_new) > 2000:
        print('Warning: input resolution is very large, results may vary')

    return w_new, h_new

def frame2tensor(frame, device):
    return torch.from_numpy(frame/255.).float()[None, None].to(device)


def read_image(path, device, resize, rotation, resize_float):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')

    if rotation != 0:
        image = np.rot90(image, k=rotation)
        if rotation % 2:
            scales = scales[::-1]

    inp = frame2tensor(image, device)
    return image, inp, scales

    
def quaternion_angular_error(q1, q2):
  """
  angular error between two quaternions
  :param q1: (4, )
  :param q2: (4, )
  :return:
  """
  d = abs(np.dot(q1, q2))
  d = min(1.0, max(-1.0, d))
  theta = 2 * np.arccos(d) * 180 / np.pi
  return theta

from .photometric import ImgAugTransform #, customizedTransform


def imgPhotometric(img, config):
    """
    :param img:
        numpy (H, W)
    :return:
    """
    augmentation = ImgAugTransform(**config)
    img = img[:,:,np.newaxis]
    img = augmentation(img)
    # cusAug = customizedTransform()
    # img = cusAug(img, **config)
    return img



def read_image_for_validation(path, device, resize, rotation, resize_float, config):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    image = image.astype(np.float32) / 255
    # cv2.imshow("thuan",image)
    if config['photometric']['enable']:
        image = imgPhotometric(image, config)
        # cv2.imshow("thuan_2",image)
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')
        
    if rotation != 0:
        image = np.rot90(image, k=rotation)
        if rotation % 2:
            scales = scales[::-1]
    inp = torch.from_numpy(image).float()[None, None].to(device)
    # inp = frame2tensor(image, device)
    return image, inp, scales
    
    
    
    
    
    
    