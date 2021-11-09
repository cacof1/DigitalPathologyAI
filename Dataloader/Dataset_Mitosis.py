# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 08:23:27 2021

@author: zhuoy
"""

import numpy as np
import time
import os
import copy
import sys
import cv2
import pandas as pd

from wsi_core.WholeSlideImage import WholeSlideImage
import geojson
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset,Subset
import torch.nn as nn
from torch.nn import functional as F
from torchvision.transforms import transforms as T

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'

dim = (256,256)
vis_level = 0

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
    
class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(self, image: Tensor,
                target: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if target is not None:
                width, _ = F._get_image_size(image)
                target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
                if "masks" in target:
                    target["masks"] = target["masks"].flip(-1)
                if "keypoints" in target:
                    keypoints = target["keypoints"]
                    keypoints = _flip_coco_person_keypoints(keypoints, width)
                    target["keypoints"] = keypoints
        return image, target
    
class ToTensor(nn.Module):
    def forward(self, image: Tensor,
                target: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.to_tensor(image)
        return image, target
    
def get_transform(train):
    transforms = []
    transforms.append(ToTensor())
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)


class Dataset(Dataset):
    def __init__(self,df, train=False):
        self.transforms = get_transform(train)
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
    
    
