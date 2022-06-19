import matplotlib.pyplot as plt
from pytorch_lightning import LightningDataModule, LightningModule, Trainer,seed_everything
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import torchvision
from Utils import GetInfo
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
import sys, glob
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping,ModelSummary
from Dataloader.Dataloader import LoadFileParameter, SaveFileParameter, DataGenerator, DataModule, WSIQuery
from Model.AutoEncoder import AutoEncoder
from pytorch_lightning.loggers import TensorBoardLogger
import toml

config   = toml.load(sys.argv[1])
##############################################
##Load File

dataset = QueryFromServer(config)
SynchronizeSVS(config, dataset)

coords_file = LoadFileParameter(config, dataset)

name   = GetInfo.format_model_name(config)
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

MasterSheet    = config['DATA']['Mastersheet']
SVS_Folder     = config['DATA']['SVS_Folder']
Patches_Folder = config['DATA']['Patches_Folder']



seed_everything(config['MODEL']['RANDOM_SEED'])
trainer   = Trainer(gpus=1, max_epochs=config['MODEL']['Max_Epochs'],precision=config['MODEL']['Precision'], callbacks = callbacks,logger=logger)
model     = AutoEncoder(config = config)

#summary(model.to('cuda'), (32, 3, 128, 128),col_names = ["input_size","output_size"],depth=5)

data      = DataModule(
    coords_file,
    batch_size       = config['MODEL']['Batch_Size'],
    train_transform  = train_transform,
    val_transform    = val_transform,
    inference        = False,
    dim_list         = config['DATA']['dim'],
    vis_list         = config['DATA']['vis'],
    n_per_sample     = config['DATA']['n_per_sample'],
    target           = config['DATA']['target'] 
)
trainer.fit(model, data)
