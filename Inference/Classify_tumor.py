# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 12:33:52 2021
@author: zhuoy/cacfek
"""
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
from Dataloader.Dataloader import LoadFileParameter, SaveFileParameter, DataModule, DataGenerator, WSIQuery

## Module - Models
from Model.ImageClassifier import ImageClassifier

# Master loader
MasterSheet      = sys.argv[1]
SVS_Folder       = sys.argv[2]
Patch_Folder     = sys.argv[3]
Pretrained_Model = sys.argv[4]

# Use same seed for now:
pl.seed_everything(42)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Local tests - please leave commented
#MasterSheet = '../__local/SarcomaClassification/data/NinjaMasterSheet.csv' # sarcoma_diagnoses.csv'  # sys.argv[1]
#SVS_Folder = '/home/mikael/Documents/data/digpath/tumor_classify_4samples/'
#Patch_Folder = '../patches/'  # sys.argv[3]
#Pretrained_Model = '../PretrainedModel/epoch=02-val_acc=1.00_tumour.ckpt'

# Current working example - with 4 specifically selected svs files.
ids = WSIQuery(MasterSheet, id='484760')
ids.extend(WSIQuery(MasterSheet, id='484761'))
ids.extend(WSIQuery(MasterSheet, id='484763'))
ids.extend(WSIQuery(MasterSheet, id='484764'))

wsi_file, coords_file = LoadFileParameter(ids, SVS_Folder, Patch_Folder)
#coords_file = coords_file[::20]  # To use a subset of the data

# Generate model
#model = ImageClassifier.load_from_checkpoint(Pretrained_Model)  # load a previous model
model = ImageClassifier(lr=1e-6, backbone=models.resnet50(pretrained=False))  # create a new one

# Training
data = DataModule(coords_file, wsi_file, train_transform=transform, val_transform=transform, batch_size=50, inference=False, dim=(256, 256), target='tumour_label')
trainer = pl.Trainer(gpus=torch.cuda.device_count(), benchmark=True, max_epochs=20, precision=16) ## Yuck but ok, it contain all the generalisation for parallel processing
res = trainer.fit(model, data)

# Make predictions
# dataset = DataLoader(DataGenerator(coords_file, wsi_file, transform=transform, inference=True), batch_size=50, num_workers= os.cpu_count(), shuffle=True, pin_memory = True)
# predictions = trainer.predict(model, dataset)
# predicted_tumour_classes_probs = torch.Tensor.cpu(torch.cat(predictions))
# for i in range(predicted_tumour_classes_probs.shape[1]):
#     SaveFileParameter(coords_file, Patch_Folder, predicted_tumour_classes_probs[:, i], 'tumour_pred_label_' + str(i))

