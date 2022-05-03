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

from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from pytorch_lightning import seed_everything
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
from Model.AutoEncoder import AutoEncoder

seed_everything(42)


##First create a master loader

MasterSheet      = sys.argv[1]
SVS_Folder       = sys.argv[2]
Patch_Folder     = sys.argv[3]
Pretrained_Model = sys.argv[4]

ids                   = WSIQuery(MasterSheet)
coords_file = LoadFileParameter(ids, SVS_Folder, Patch_Folder)
coords_file = coords_file[coords_file["tumour_label"] == 1]

transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),    
])

invTrans   = transforms.Compose([
    #transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1./0.229, 1./0.224, 1./0.225 ]),
    #transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]),
    torchvision.transforms.ToPILImage()
    ])

## Load the previous  model
trainer = pl.Trainer(gpus=torch.cuda.device_count(), benchmark=True, max_epochs=20, precision=32)
trainer.model = AutoEncoder.load_from_checkpoint(Pretrained_Model)
output_shape = trainer.model.predict(torch.rand((1, n_classes, dim[0], dim[1]))).size()
print("Inference",output_shape)
## Now train
test_dataset = DataLoader(DataGenerator(coords_file[:1000], transform = transform, inference = True), batch_size=10, num_workers=0, shuffle=False)
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

