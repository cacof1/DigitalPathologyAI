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
import h5py
import sys

from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import savePatchIter_bag_hdf5, initialize_hdf5_bag, coord_generator, save_hdf5, sample_indices, screen_coords, isBlackPatch, isWhitePatch, to_percentiles
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
from Dataloader.Dataloader import LoadFileParameter, DataModule, DataGenerator

## Module - Models
from Model.TumorClassifier import ImageClassifier

##First create a master loader

wsi_file, coords_file = LoadFileParameter(sys.argv[1])

## Load the previous  model
model = ImageClassifier.load_from_checkpoint(sys.argv[2])

## Now train
trainer = pl.Trainer(gpus=1) ## Yuck
dataset = DataLoader(DataGenerator(coords_file, wsi_file), batch_size=80, num_workers=10)
preds   = trainer.predict(model,dataset)
print(coords_file)
print(len(preds))

## Now we save to hdf5
Path("Preprocessing/patches_tumor").mkdir(parents=True, exist_ok=True)
