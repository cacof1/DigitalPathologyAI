# -*- coding: utf-8 -*-
"""
Created on Mon May 16 11:55:11 2022

@author: zhuoy
"""

import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
import openslide
from torchvision.transforms import functional as F
from Utils.ObjectDetectionTools import collate_fn

import torch
from torch import nn, Tensor
from torch.utils.data import Dataset,DataLoader
from typing import Tuple, Dict, Optional
from pytorch_lightning import LightningDataModule

def get_bbox_from_mask(mask):
    pos = np.where(mask==255)
    if pos[0].shape[0] == 0:
        return np.zeros((0, 4))
    else:
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        return [xmin, ymin, xmax, ymax]

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor(nn.Module):
    def forward(self, image: Tensor,
                target: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.to_tensor(image)
        return image, target
    
transform = Compose([ToTensor(),
                     ])

class MFDataset(Dataset):

    def __init__(self, 
                 df,
                 wsi_folder,
                 mask_folder,
                 augmentation=None, 
                 normalization=None,
                 inference=False,
                 ):

        self.df = df
        self.wsi_folder = wsi_folder
        self.mask_folder = mask_folder
        self.transform = Compose([ToTensor(),
                             ])
        
        self.augmentation = augmentation
        self.normalization = normalization
        self.inference = inference

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, i):
        
        vis_level = 0
        dim = (256,256)
        index = self.df['index'][i]
        filename = self.df['filename'][i]        
        top_left = (self.df['mitosis_coord_x'][i],self.df['mitosis_coord_y'][i])
        wsi_object = openslide.open_slide(self.wsi_folder+'{}.svs'.format(filename))
        img = np.array(wsi_object.read_region(top_left, vis_level, dim).convert("RGB"))
        
        num_objs = self.df['num_objs'][i]  
        
        if num_objs == 0:
            mask = np.zeros((dim[0], dim[1]))         
        else:         
            mask = np.load(self.mask_folder+'{}_masks.npy'.format(filename))[index]
            
        if self.augmentation is not None:
            transformed = self.augmentation(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]
            
        masks = mask[np.newaxis,:, :] 
        boxes = []  
        area = []
        
        for n in range(num_objs):
            box = get_bbox_from_mask(masks[n])
            boxes.append(box)
            area.append((box[2] - box[0]) * (box[3] - box[1]))
            
        obj_ids = np.array([255])     
        masks = mask == obj_ids[:, None, None]
        
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
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
            img, target = self.transform(img, target)
            
        if self.normalization is not None:
            img = self.normalization(img)
            
        if self.inference:
            return img
        else:
            return img, target
      
class MFDataModule(LightningDataModule):
    def __init__(self, 
                 df_train,
                 df_val,
                 df_test,
                 wsi_folder,
                 mask_folder,
                 batch_size=2,
                 num_of_worker=0,
                 augmentation=None, 
                 normalization=None,
                 inference=False,):
        
        super().__init__()
        self.batch_size = batch_size
        self.num_of_worker = num_of_worker
        self.train_data = MFDataset(df_train,
                                    wsi_folder,
                                    mask_folder,
                                    augmentation=augmentation,
                                    normalization=normalization,
                                    inference=inference)
        
        self.val_data = MFDataset(df_val,
                                  wsi_folder,
                                  mask_folder,
                                  augmentation=None,
                                  normalization=normalization,
                                  inference=inference)
        
        self.test_data = MFDataset(df_test,
                                   wsi_folder,
                                   mask_folder,
                                   augmentation=None,
                                   normalization=normalization,
                                   inference=inference)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_of_worker, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_of_worker, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=1, shuffle=False, num_workers=self.num_of_worker, collate_fn=collate_fn)
