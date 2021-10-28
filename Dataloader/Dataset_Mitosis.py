# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 08:23:27 2021

@author: zhuoy
"""

import numpy as np
import time
import os
import copy
import h5py
import sys
import cv2
import pandas as pd

from wsi_core.WholeSlideImage import WholeSlideImage
import geojson
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset,Subset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'

dim = (256,256)
vis_level = 0

class Dataset(Dataset):
    def __init__(self,df, transforms):
        self.transforms = transforms
        self.df = df

    def __getitem__(self, i):
        # load images and masks
        vis_level = 0
        dim = (256,256)
        
        filename = self.df['filename'][i]
        wsi_object = WholeSlideImage(basepath + 'data/wsi/{}.svs'.format(filename))
        top_left = (self.df['top_left_x'][i],self.df['top_left_y'][i])
        img = np.array(wsi_object.wsi.read_region(top_left, vis_level, dim).convert("RGB"))
        num_objs = 1
        boxes = []
        labels = []
        for n in range(num_objs):
            xmin = self.df.x_min[i]-top_left[0]
            xmax = self.df.x_max[i]-top_left[0]
            ymin = self.df.y_min[i]-top_left[1]
            ymax = self.df.y_max[i]-top_left[1]
            box = [xmin, ymin, xmax, ymax]
            label = self.df.labels[i]
            boxes.append(box)
            labels.append(label)

        #cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(1))
        #plt.imshow(img)    
        #plt.show() 
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([i])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target['area'] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
            
        return img, target,image_id

    def __len__(self):
        return self.df.shape[0]
    
    
