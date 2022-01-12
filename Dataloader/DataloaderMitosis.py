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
from torchvision import datasets, models
import torchvision
import transforms as T
from engine import train_one_epoch, evaluate
from StainNorm import normalizeStaining


class DataGenerator_Mitosis(torch.utils.data.Dataset):
    def __init__(self,df, wsi_path, mask_path,transform=None):
        self.transforms = transforms
        self.df = df
        self.normalizer = TorchMacenkoNormalizer()
        self.transform = transform

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

        if self.transform is not None:
            img, target = self.transform(img,target)
            
        return img, target

    def __len__(self):
        return self.df.shape[0]
    
    
    
class DataModule_Mitosis(LightningDataModule):
    def __init__(self, mitosis_file, wsi_path, mask_path, batch_size = 1, train_transform = None, val_transform = None,  **kwargs):
        super().__init__()
        self.batch_size      = batch_size        
          
        ids_split            = np.round(np.array([0.7, 0.2, 1.0])*len(mitosis_file)).astype(np.int32)
        self.train_data      = DataGenerator_MitotsisDetection(mitosis_file[ids_split[0]:ids_split[1]], wsi_path, mask_path, transform = train_transform, **kwargs)
        self.val_data        = DataGenerator_MitotsisDetection(mitosis_file[ids_split[0]:ids_split[1]], wsi_path, mask_path,  transform = val_transform, **kwargs)
        self.test_data       = DataGenerator_MitotsisDetection(mitosis_file[ids_split[0]:ids_split[1]], wsi_path, mask_path, transform = val_transform, **kwargs)


    def train_dataloader(self): return DataLoader(self.train_data, batch_size=self.batch_size,num_workers=0)
    def val_dataloader(self):   return DataLoader(self.val_data, batch_size=self.batch_size,num_workers=0)
    def test_dataloader(self):  return DataLoader(self.test_data, batch_size=self.batch_size)
