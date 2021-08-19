# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 00:08:35 2021

@author: zhuoy
"""

import os
import cv2
import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from wsi_core.WholeSlideImage import WholeSlideImage
from torch.utils.data import DataLoader
from torch.utils.data import Dataset,Subset
from torchvision import datasets, models,transforms
import matplotlib.pyplot as plt


basepath = sys.argv[1]
filename = sys.argv[2]

coords_file = h5py.File(basepath + 'wsi/{}.h5'.format(filename),'r')
wsi_object = WholeSlideImage(basepath + 'wsi/{}.svs'.format(filename))
coords = coords_file['coords']   

vis_level = 0
dim = (256,256)

dirs = os.listdir(basepath + '{}_mitosis/'.format(filename))
ids = []
for i in dirs:
    if os.path.splitext(i)[1] == '.npz':
        ids.append(i)
    

def visualize_prediction(file):
    data = np.load(basepath + '{}_mitosis/{}'.format(filename,file))
    top_left = data['top_left']
    boxes = data['boxes']
    scores = data['scores']
    img = np.array(wsi_object.wsi.read_region(top_left, vis_level, dim).convert("RGB"))
    count = 0
    for i in range(scores.shape[0]):
        box = boxes[i]
        score = scores[i]
        xmin = int(box[0])
        ymin = int(box[1])
        xmax = int(box[2])
        ymax = int(box[3])
        if score > 0.5:
            cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(255*score,255,0),thickness=2)
    
    if scores.shape[0] != 0:
        if max(scores) > 0.7:
            plt.imshow(img) 
            plt.title('{} top left: ({},{})'.format(filename,top_left[0],top_left[1]))
            plt.show() 
            count = 1
            
    return count
    

counts = 0
for file in ids:
    counts += visualize_prediction(file)
    
print(counts)
    
