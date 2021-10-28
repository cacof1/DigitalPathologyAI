# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 08:23:27 2021

@author: zhuoy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import time
import os
import copy
import h5py
import sys
import cv2
import pandas as pd
import transforms as T
from engine import train_one_epoch, evaluate
import utils_

from wsi_core.WholeSlideImage import WholeSlideImage
import geojson
from torchvision import transforms
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset,Subset
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'

dim = (256,256)
vis_level = 0

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

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
    
    
