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

callbacks = [
    ModelCheckpoint(
        dirpath='./',
        monitor='val_loss',
        filename="model_AutoEncoder",#.{epoch:02d}-{val_loss:.2f}.h5",
        save_top_k=1,
        mode='min'),
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
#coords_file = coords_file[:40000]
seed_everything(42)                                                                                                                                                                                           

trainer   = Trainer(gpus=1, max_epochs=15,precision=16, callbacks = callbacks)
model     = AutoEncoder()

dim       = (96,96)
vis_level = 0
data      = DataModule(coords_file, batch_size=64, train_transform = train_transform, val_transform = val_transform, inference=False, dim=dim, vis_level = vis_level)
trainer.fit(model, data)

## Testing                                                                                                                                                                                                     
test_dataset       = DataGenerator(coords_file, inference = True,transform=val_transform, dim=dim, vis_level = vis_level)
n = 10
plt.figure(figsize=(20, 4))

for i in range(n):
    idx        = np.random.randint(len(coords_file),size=1)[0]
    image     = test_dataset[idx][np.newaxis]
    image_out = trainer.model.forward(image)

    image     = image.squeeze().detach().cpu().numpy()#*255.
    image     = image.transpose((1, 2,0))#.astype('uint8')

    image_out = image_out.squeeze().detach().cpu().numpy()*255.
    image_out = image_out.transpose((1,2,0))#.astype('uint8')

    #image     = invTrans(image.squeeze())
    #image_out = invTrans(image_out.squeeze())
    #print(type(image))
    #print(type(image_out))
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(image)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(image_out)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
plt.show()
