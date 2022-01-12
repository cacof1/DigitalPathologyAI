# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 10:24:27 2021

@author: zhuoy
"""

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
import math
import pandas as pd
from wsi_core.WholeSlideImage import WholeSlideImage
from torch.utils.data import DataLoader
from torch.utils.data import Dataset,Subset
from torchvision import datasets, models
import torchvision

import transforms as T
from engine import train_one_epoch, evaluate
import utils_
from StainNorm import normalizeStaining

class MaskDataset(Dataset):
    def __init__(self,df, transforms,wsi_path,mask_path):
        self.transforms = transforms
        self.df = df

    def __getitem__(self, i):
        # load images and masks
        vis_level = 0
        dim = (256,256)
                
        index = self.df['index'][i]
        
        filename = self.df['filename'][i]        
        top_left = (self.df['mitosis_coord_x'][i],self.df['mitosis_coord_y'][i])
        
        wsi_object = WholeSlideImage(wsi_path + '/{}.svs'.format(filename))  

        img = np.array(wsi_object.wsi.read_region(top_left, vis_level, dim).convert("RGB"))
        num_objs = self.df['num_objs'][i]  
        
        try:
            img,H,E = normalizeStaining(img)
        except:
            pass
            
        if num_objs == 0:
            mask = np.zeros((dim[0], dim[1]))
            obj_ids = np.unique(mask)[:]
            masks = mask != obj_ids[:, None, None]
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            area = [0]
            
        else:         
            mask = np.load(mask_path + '/{}_mitosis_masks.npy'.format(filename))[index]
            masks = mask[np.newaxis,:, :]
            boxes = []  
            area = []
            
            for n in range(num_objs):
                pos = np.where(masks[n]==255)
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])
                area.append((xmax - xmin) * (ymax - ymin))
            
            boxes = torch.as_tensor(boxes, dtype=torch.float32)

            obj_ids = np.unique(mask)[1:]       
            masks = mask == obj_ids[:, None, None]
        
        masks = torch.as_tensor(masks, dtype=torch.uint8)
            
        labels = torch.ones((num_objs,), dtype=torch.int64)   
        area = torch.as_tensor(area, dtype=torch.float32)
        
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([i])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target['area'] = area
        target["iscrowd"] = iscrowd        
        target["masks"] = masks

        if self.transforms is not None:
            img, target = self.transforms(img, target)
            
        return img, target

    def __len__(self):
        return self.df.shape[0]
