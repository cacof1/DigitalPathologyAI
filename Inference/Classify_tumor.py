# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 12:33:52 2021
@author: zhuoy/cacfek
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import sys

from wsi_core.WholeSlideImage import WholeSlideImage
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.nn import functional as F
from torch.nn.functional import softmax
from pytorch_lightning.callbacks import ModelCheckpoint

from pathlib import Path

## Module - Dataloaders
from Dataloader.Dataloader import LoadFileParameter, SaveFileParameter, DataGenerator, WSIQuery

## Module - Models
from Model.TumorClassifier import ImageClassifier

##First create a master loader

MasterSheet      = sys.argv[1]
SVS_Folder       = sys.argv[2]
Patch_Folder     = sys.argv[3]
Pretrained_Model = sys.argv[4]

ids                   = WSIQuery(MasterSheet)
wsi_file, coords_file = LoadFileParameter(ids, SVS_Folder, Patch_Folder)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),    
])

## Load the previous  model
model = ImageClassifier.load_from_checkpoint(Pretrained_Model)

## Now train

trainer = pl.Trainer(accelerator='auto', benchmark = True) ## Yuck but ok, it contain all the generalisation for parallel processing
dataset = DataLoader(DataGenerator(coords_file, wsi_file, transform = transform, inference = True), batch_size=10, num_workers=10, shuffle=False)
preds   = trainer.predict(model,dataset)
predictions = []
for i in range(len(preds)):
    indpred = preds[i][0].tolist()[0][1]
    predictions.append(indpred)

SaveFileParameter(coords_file, Patch_Folder, predictions,"tumour_label")




