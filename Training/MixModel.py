import matplotlib.pyplot as plt
from pytorch_lightning import LightningDataModule, LightningModule, Trainer,seed_everything
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models, transforms
from torchvision import transforms
from torchinfo import summary
import pandas as pd
import cv2
from torch.utils.data import Dataset
from pytorch_lightning.loggers import TensorBoardLogger
import torchvision.models as models
import numpy as np
import torch
from torchinfo import summary

import openslide
import sys, glob
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from wsi_core.WholeSlideImage import WholeSlideImage
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping,ModelSummary
from Dataloader.Dataloader import LoadFileParameter, SaveFileParameter, DataGenerator, DataModule, WSIQuery
from Model.MixModel import MixModel
from Model.ImageClassifier import ImageClassifier
from pytorch_lightning.loggers import TensorBoardLogger

checkpoint_callback = ModelCheckpoint(
    dirpath='./',
    monitor='val_loss',
    filename="MixModel",#.{epoch:02d}-{val_loss:.2f}.h5",
    save_top_k=1,
    mode='min')

callbacks = [
    checkpoint_callback
    ]
train_transform = transforms.Compose([
    transforms.ToTensor(),
])

val_transform   = transforms.Compose([
    transforms.ToTensor(),
])


logger         = TensorBoardLogger('lightning_logs',name = 'test')
MasterSheet    = sys.argv[1]
SVS_Folder     = sys.argv[2]
Patches_Folder = sys.argv[3]

ids           = WSIQuery(MasterSheet)
coords_file   = LoadFileParameter(ids, SVS_Folder, Patches_Folder)
#coords_file   = coords_file[coords_file["tumour_label"] == 1]
seed_everything(42)                                                                                                                                                                                           

trainer   = Trainer(gpus=1, max_epochs=5,precision=32, callbacks = callbacks,logger=logger)

dim_list = [(128,128), (128,128)]
vis_list = [0,1]
keys     = []
for dim in dim_list:
            for vis_level in vis_list:
                keys.append("_".join(map(str,dim))+"_"+str(vis_level))

model_dict  = nn.ModuleDict({
    keys[0]: ImageClassifier().backbone,
    keys[1]: ImageClassifier().backbone    
})
model     = MixModel(model_dict)
data      = DataModule(coords_file, batch_size=32, train_transform = train_transform, val_transform = val_transform, inference=False, dim_list=dim_list, vis_list = vis_list)
trainer.fit(model, data)

