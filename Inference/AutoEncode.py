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
import toml

## Module - Dataloaders
from Dataloader.Dataloader import *

## Module - Models
from Model.AutoEncoder import AutoEncoder

config   = toml.load(sys.argv[1])
seed_everything(42)


##First create a master loader
SVS_dataset = QueryFromServer(config)
SynchronizeSVS(config, SVS_dataset)
DownloadNPY(config, SVS_dataset)
tile_dataset = LoadFileParameter(config, SVS_dataset)
tile_dataset = tile_dataset[tile_dataset['prob_tissue_type_tumour'] > 0.85]
Pretrained_Model = sys.argv[2]

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
trainer = pl.Trainer(gpus=torch.cuda.device_count(), precision=config['BASEMODEL']['Precision'], benchmark=True, max_epochs=config['BASEMODEL']['Max_Epochs'])
trainer.model = AutoEncoder.load_from_checkpoint(Pretrained_Model, config=config)

print(tile_dataset)
## Now test
test_dataset = DataLoader(DataGenerator(tile_dataset[:100],
                                        transform  = transform,
                                        inference  = True,
                                        svs_folder = config['DATA']['SVS_Folder']),
                          batch_size  = config['BASEMODEL']['Batch_Size'],
                          num_workers = 0,
                          shuffle     = False)

image_out    = trainer.predict(trainer.model,test_dataset)

batch_size = config['BASEMODEL']['Batch_Size']
tmp = iter(test_dataset)
print(len(tmp))
for j in range(n):
    plt.figure(figsize=(20, 4))
    batch= next(tmp)
    for i in range(batch_size):
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

