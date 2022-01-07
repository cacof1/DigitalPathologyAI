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
from Model.AutoEncoder import AutoEncoder
from pytorch_lightning.loggers import TensorBoardLogger
checkpoint_callback = ModelCheckpoint(
    dirpath='./',
    monitor='val_loss',
    filename="model_AutoEncoder",#.{epoch:02d}-{val_loss:.2f}.h5",
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

invTrans   = transforms.Compose([
    torchvision.transforms.ToPILImage()
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
model     = AutoEncoder()


summary(model.to('cuda'), (32, 3, 128, 128),col_names = ["input_size","output_size"],depth=5)
dim       = (128,128)
vis_level = 0
data      = DataModule(coords_file, batch_size=32, train_transform = train_transform, val_transform = val_transform, inference=False, dim=dim, vis_level = vis_level)
trainer.fit(model, data)


## Testing
test_dataset = DataLoader(DataGenerator(coords_file[:1000], transform = train_transform, inference = True), batch_size=10, num_workers=0, shuffle=False)
image_out    = trainer.predict(trainer.model,test_dataset)
n = 10
tmp = iter(test_dataset)
for j in range(n):
    plt.figure(figsize=(20, 4))
    image = next(tmp)
    for i in range(n):
        img      = invTrans(image[i])
        img_out  = invTrans(image_out[j][i])
        ax = plt.subplot(2, n, i + 1)
        if(i==0):ax.set_title("image_in")
        plt.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        ax = plt.subplot(2, n, i + 1 + n)
        if(i==0):ax.set_title("image_out")
        plt.imshow(img_out)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
