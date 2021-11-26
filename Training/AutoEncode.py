import matplotlib.pyplot as plt
from pytorch_lightning import LightningDataModule, LightningModule, Trainer,seed_everything
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models, transforms
from torchvision import transforms
from torchsummary import summary
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
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from wsi_core.WholeSlideImage import WholeSlideImage
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping,ModelSummary
from Dataloader.Dataloader import LoadFileParameter, SaveFileParameter, DataGenerator, DataModule, WSIQuery
from Model.AutoEncoder import AutoEncoder

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
    #transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform   = transforms.Compose([
    #transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

invTrans   = transforms.Compose([
    #transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1./0.229, 1./0.224, 1./0.225 ]),
    #transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]),
    #torchvision.transforms.ToPILImage()
])

MasterSheet    = sys.argv[1]
SVS_Folder     = sys.argv[2]
Patches_Folder = sys.argv[3]

ids = WSIQuery(MasterSheet)

coords_file = LoadFileParameter(ids, SVS_Folder, Patches_Folder)
coords_file = coords_file[:20000]
seed_everything(42)                                                                                                                                                                                           

trainer   = Trainer(gpus=1, max_epochs=1,precision=32, callbacks = callbacks)
model     = AutoEncoder()

dim       = (96,96)
vis_level = 0
data      = DataModule(coords_file, batch_size=64, train_transform = train_transform, val_transform = val_transform, inference=False, dim=dim, vis_level = vis_level)
trainer.fit(model, data)

## Testing
test_dataset = DataLoader(DataGenerator(coords_file[:100], transform = transform, inference = True), batch_size=10, num_workers=0, shuffle=False)

image      = next(iter(test_dataset))
image_out  = trainer.predict(trainer.model,test_dataset)
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    img      = invTrans(image[i])
    img_out  = invTrans(image_out[0][i])
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
