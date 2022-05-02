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
import openslide
from torch import Tensor
import math
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets, models
import torchvision
import transforms as T
from engine import train_one_epoch, evaluate
from StainNorm import normalizeStaining


class DataGenerator_Mitosis(torch.utils.data.Dataset):
    def __init__(self,
                 df, 
                 mask_path=None,
                 region_level = 'patch',#or 'cell'
                 transforms=None,
                 augmentation=None, 
                 predicting=False,
                 inference=False):
        
        self.df = df
        self.mask_path = mask_path
        self.region_level = region_level
        self.transforms = transforms        
        self.augmentation = augmentation
        self.predicting = predicting
        self.inference = inference

    def __getitem__(self, i):
        
        vis_level = 0
        dim = (256,256)
       
        wsi_path = self.df['wsi_path']
        wsi_object = openslide.open_slide(wsi_path)  
        img = np.array(wsi_object.read_region(top_left, vis_level, dim).convert("RGB"))
        try:
            img,H,E = normalizeStaining(img)
        except:
            pass
        
        if self.predicting:
            if self.transforms:
                img = self.transforms(img)
                
            return img
        
        index = self.df['index'][i]
        filename = self.df['filename'][i]        
        top_left = (self.df['mitosis_coord_x'][i],self.df['mitosis_coord_y'][i])
        num_objs = self.df['num_objs'][i]  
        
        mask = np.load(self.mask_path + '/{}_masks.npy'.format(filename))[index]
        pos = np.where(mask==255)
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        box = [xmin, ymin, xmax, ymax]
        area = (xmax - xmin) * (ymax - ymin)
        
        center = (int(0.5*(xmax+xmin)),int(0.5*(ymax+ymin)))
        
        x1 = int(center[0]-32)
        x2 = int(center[0]+32)
        y1 = int(center[1]-32)
        y2 = int(center[1]+32)
        
        if x1 < 0:
            x1 = 0
            x2 = x1 + 64
        if x2 > 256:
            x2 = 256
            x1 = x2 - 64
        if y1 < 0:
            y1 = 0
            y2 = y1 + 64
        if y2 >256:
            y2 = 256
            y1 = y2 - 64
                
        region = (x1,x2,y1,y2) 
                                        
        cell = img[region[2]:region[3],region[0]:region[1]]
        cell_mask = mask[region[2]:region[3],region[0]:region[1]]

        obj_ids = np.unique(mask)[1:]       
        mask = mask == obj_ids[:, None, None]
        obj_ids = np.unique(cell_mask)[1:]       
        cell_mask = cell_mask == obj_ids[:, None, None]
        
        labels = torch.ones((num_objs,), dtype=torch.int64)   
        area = torch.as_tensor(area, dtype=torch.float32)
        box = torch.as_tensor(box, dtype=torch.float32)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([i])

        target = {}
        target["boxes"] = box
        target["labels"] = labels
        target["image_id"] = image_id
        target['area'] = area
        target["iscrowd"] = iscrowd        
        
        if self.region_level == 'patch':
            
            if self.augmentation:
                img, mask = self.augmentation(img, mask)
                
            mask = torch.as_tensor(mask, dtype=torch.uint8)
            target["masks"] = mask
            
            if self.transforms:
                img, target = self.transforms(img, target)
                
            if self.inference:
                return img
                             
            return img, target     
            
        if self.region_level == 'cell':
            
            if self.augmentation:
                cell,mask = self.augmentation(cell,mask)
        
            if self.transforms:               
                sample = self.transforms(image=cell, mask=cell_mask)
                cell = torch.as_tensor(sample['image'], dtype=torch.float32)
                cell_mask = sample['mask'][np.newaxis,:, :]
                cell_mask = torch.as_tensor(cell_mask, dtype=torch.float32)
                     
            if self.inference:
                return cell
            
            return cell, cell_mask 
            
    def __len__(self):
        return self.df.shape[0]
    
    
    
class DataModule_Mitosis(LightningDataModule):
    def __init__(self, mitosis_file, mask_path, batch_size = 1, train_transform = None, val_transform = None,  **kwargs):
        super().__init__()
        self.batch_size      = batch_size        
          
        ids_split            = np.round(np.array([0.7, 0.2, 1.0])*len(mitosis_file)).astype(np.int32)
        self.train_data      = DataGenerator_MitotsisDetection(mitosis_file[ids_split[0]:ids_split[1]], mask_path, transform = train_transform, **kwargs)
        self.val_data        = DataGenerator_MitotsisDetection(mitosis_file[ids_split[0]:ids_split[1]], mask_path,  transform = val_transform, **kwargs)
        self.test_data       = DataGenerator_MitotsisDetection(mitosis_file[ids_split[0]:ids_split[1]], mask_path, transform = val_transform, **kwargs)


    def train_dataloader(self): return DataLoader(self.train_data, batch_size=self.batch_size,num_workers=0)
    def val_dataloader(self):   return DataLoader(self.val_data, batch_size=self.batch_size,num_workers=0)
    def test_dataloader(self):  return DataLoader(self.test_data, batch_size=self.batch_size)
