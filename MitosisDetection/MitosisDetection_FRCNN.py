# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 11:22:24 2021

@author: zhuoy
"""
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
import pandas as pd
from wsi_core.WholeSlideImage import WholeSlideImage
from torch.utils.data import DataLoader
from torch.utils.data import Dataset,Subset
from torchvision import datasets, models
import transforms as T
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from engine import train_one_epoch, evaluate
import utils_


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

class Dataset(Dataset):
    def __init__(self,df,wsi_object,transforms):
        self.transforms = transforms

        self.df = df
        self.wsi = wsi_object
    def __getitem__(self, i):
        # load images and masks
        vis_level = 0
        dim = (256,256)
        top_left = (df['top_left_x'][i],df['top_left_y'][i])
        img = np.array(wsi_object.wsi.read_region(top_left, vis_level, dim).convert("RGB"))
        num_objs = 1
        boxes = []
        for n in range(num_objs):
            xmin = df.x_min[i]-top_left[0]
            xmax = df.x_max[i]-top_left[0]
            ymin = df.y_min[i]-top_left[1]
            ymax = df.y_max[i]-top_left[1]
            box = [xmin, ymin, xmax, ymax]
            boxes.append(box)

        #cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(1))
        #plt.imshow(img)    
        #plt.show() 

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        label = torch.ones((num_objs,), dtype=torch.int64)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        image_id = torch.tensor([i])

        target = {}
        target["boxes"] = boxes
        target["labels"] = label
        target["image_id"] = image_id
        target['area'] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
            
        return img, target

    def __len__(self):
        return self.df.shape[0]


def get_model_instance_segmentation(num_classes):

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 2

basepath = sys.argv[1]
filename = sys.argv[2]
wsi_object = WholeSlideImage(basepath + 'wsi/{}.svs'.format(filename))
df = pd.read_csv('mitosis{}.csv'.format(filename))                 
dataset = Dataset(df,wsi_object,get_transform(train=True))
dataset_test = Dataset(df,wsi_object,get_transform(train=False))
indices = torch.randperm(len(dataset)).tolist()
dataset_train = Subset(dataset, indices[:-10])
dataset_test = Subset(dataset_test, indices[-10:])

data_loader_train = DataLoader(
        dataset_train, batch_size=1, shuffle=True, num_workers=0,
        collate_fn=utils_.collate_fn)

data_loader_test = DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils_.collate_fn)

model = get_model_instance_segmentation(num_classes)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

num_epochs = 10

for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
    lr_scheduler.step()
    evaluate(model, data_loader_test, device=device)
    
torch.save(model, 'fasterrcnn_resnet50_fpn')
#torch.save(model.state_dict(), 'model_weights')
