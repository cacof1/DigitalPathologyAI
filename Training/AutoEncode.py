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
import toml



config   = toml.load(sys.argv[1])
name     = config['MODEL']['BaseModel'] +"_"+ config['MODEL']['Backbone']+ "_wf" + str(config['MODEL']['wf']) + "_depth" + str(config['MODEL']['depth'])
logger   = TensorBoardLogger('lightning_logs',name = name)
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

ids           = WSIQuery(MasterSheet, config)
coords_file   = LoadFileParameter(ids, SVS_Folder, Patches_Folder)
#coords_file   = coords_file[coords_file["tumour_label"] == 1]


seed_everything(config['MODEL']['RANDOM_SEED'])
trainer   = Trainer(gpus=1, max_epochs=config['MODEL']['Max_Epochs'],precision=config['MODEL']['Precision'], callbacks = callbacks,logger=logger)
model     = AutoEncoder(config = config)

summary(model.to('cuda'), (32, 3, 128, 128),col_names = ["input_size","output_size"],depth=5)

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


## Testing
test_dataset = DataLoader(DataGenerator(coords_file[:1000], transform = train_transform, inference = True), batch_size=10, num_workers=0, shuffle=False)
image_out    = trainer.predict(trainer.model,test_dataset)
n = 10
tmp = iter(test_dataset)
for j in range(n):
    plt.figure(figsize=(20, 4))
    image  = next(tmp)
    image  = next(iter(image.values())) 
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
