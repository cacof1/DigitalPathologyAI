import matplotlib.pyplot as plt
from pytorch_lightning import LightningDataModule, LightningModule, Trainer,seed_everything
from torch.utils.data import DataLoader
import torchvision
from Utils import GetInfo
from torchvision import datasets, models, transforms
from torchvision import transforms
import pandas as pd
import cv2
from torch.utils.data import Dataset
from pytorch_lightning.loggers import TensorBoardLogger
import torchvision.models as models
import numpy as np
import torch
import sys, glob
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping,ModelSummary
from Dataloader.Dataloader import *
from Model.AutoEncoder import AutoEncoder
from pytorch_lightning.loggers import TensorBoardLogger

import toml

config   = toml.load(sys.argv[1])
name     = "AutoEncoder"#GetInfo.format_model_name(config)
##############################################
##Load File

SVS_dataset = QueryFromServer(config)
SynchronizeSVS(config, SVS_dataset)
DownloadNPY(config, SVS_dataset)

tile_dataset = LoadFileParameter(config, SVS_dataset)

logger = TensorBoardLogger('lightning_logs',name = name)
checkpoint_callback = ModelCheckpoint(
    dirpath     = logger.log_dir,
    monitor     = 'val_loss',
    filename    = name,
    save_top_k  = 1,
    mode        = 'min')

callbacks = [checkpoint_callback]

train_transform = transforms.Compose([
    transforms.ToTensor(),
])

val_transform   = transforms.Compose([
    transforms.ToTensor(),
])

invTrans   = transforms.Compose([
    torchvision.transforms.ToPILImage()
])

seed_everything(config['ADVANCEDMODEL']['Random_Seed'])
trainer   = Trainer(gpus=1, max_epochs=config['BASEMODEL']['Max_Epochs'],precision=config['BASEMODEL']['Precision'], callbacks = callbacks,logger=logger)
model     = AutoEncoder(config)

data = DataModule(
    tile_dataset,
    batch_size      = config['BASEMODEL']['Batch_Size'],
    train_transform = train_transform,
    val_transform   = val_transform,
    train_size      = config['DATA']['Train_Size'],
    val_size        = config['DATA']['Val_Size'],
    inference       = False,
    dim             = config['BASEMODEL']['Patch_Size'],
    vis             = config['BASEMODEL']['Vis'],
    n_per_sample    = config['DATA']['N_Per_Sample'],
    target          = config['DATA']['Label'],
    sampling_scheme = config['DATA']['Sampling_Scheme'],
    svs_folder      = config['DATA']['SVS_Folder'],
)

trainer.fit(model, data)
