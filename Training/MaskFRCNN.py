import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models, transforms
from torchvision import transforms
import pandas as pd
import cv2
from torch.utils.data import Dataset
from pytorch_lightning.loggers import TensorBoardLogger
import torchvision.models as models
import numpy as np
import torch
import openslide
import sys, glob
import torch.nn as nn
from wsi_core.WholeSlideImage import WholeSlideImage
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping,ModelSummary
from Dataloader.Dataloader import LoadFileParameter, SaveFileParameter, DataGenerator, DataModule, WSIQuery
from pytorch_lightning.loggers import TensorBoardLogger
import toml
from Dataloader.DataloaderMitosis import DataGenerator,DataGenerator_Mitosis
from Model.MaskFRCNN import MaskFRCNN
import transforms as T

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

config   = toml.load(sys.argv[1])
name     = config['MODEL']['ModelName'] 
logger   = TensorBoardLogger('lightning_logs',name = name)
checkpoint_callback = ModelCheckpoint(
    dirpath     = logger.log_dir,
    monitor     = 'val_iou',
    filename    = name,
    save_top_k  = 1,
    mode        = 'max')

callbacks = [checkpoint_callback]

df = pd.read_csv(config['Data']['MitosisFile'] )
df_all = df[df['num_objs']==1]
df_all = df_all[df_all['QA']==1]
filenames = df_all.filename.value_counts().index.to_list()

filenames_test = config['Data']['Testing']
filenames_val = config['Data']['Validating']

df_train = df_all[~df_all['filename'].isin(filenames_test+filenames_val)]
df_train.reset_index(drop = True, inplace = True)

df_val = df_all[df_all['filename'].isin(filenames_val)]
df_val.reset_index(drop = True, inplace = True)

df_test = df_all[df_all['filename'].isin(filenames_test)]
df_test.reset_index(drop = True, inplace = True)

dataset_train = DataGenerator_Mitosis(df_train,get_transform(train=True))
dataset_val = DataGenerator_Mitosis(df_val,get_transform(train=False))

train_size = len(dataset_train)
val_size = len(dataset_val)
test_size = df_test.shape[0]
print('Training Size: {}, Validating Size: {}, Testing Size: {}'.format(train_size,
                                                                       val_size,
                                                                       test_size))

data_loader_train = DataLoader(
        dataset_train, batch_size=2, shuffle=True, num_workers=0)

data_loader_val = DataLoader(
        dataset_val, batch_size=1, shuffle=False, num_workers=0)

trainer   = pl.Trainer(gpus=1, max_epochs=config['MODEL']['Max_Epochs'],precision=config['MODEL']['Precision'], callbacks = callbacks,logger=logger)
model     = MaskFRCNN(num_classes=2,lr=0.0025)
trainer.fit(model, 
            train_dataloaders=data_loader_train, 
            val_dataloaders=data_loader_val,
            )

valid_metrics = trainer.validate(model, dataloaders=data_loader_val, verbose=False)
