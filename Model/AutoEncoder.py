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
import h5py, sys, glob
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from wsi_core.WholeSlideImage import WholeSlideImage
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

## Module - Utils
from utils.StainNorm import normalizeStaining

## Module - Dataloaders
from Dataloader.Dataloader import LoadFileParameter, DataModule, DataGenerator

## Module - Models
from Model.unet import UNet
from Model.resnet import ResNet, ResNetResidualBlock

## Main Model
class AutoEncoder(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        
        self.model = UNet(in_channels=3, n_classes = 3)
        #self.model =  ResNet(in_channels=3, n_classes=3, block=ResNetResidualBlock)
        self.loss_fcn = torch.nn.MSELoss(reduction="sum")
        #self.loss_fcn = torch.nn.L1Loss(reduction="sum")        

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        #summary(self.model, (3,256,256))                
        return self.model(x)
    
   
    def training_step(self, batch, batch_idx):
        image,label = batch
        prediction  = self.forward(image)
        loss = self.loss_fcn(prediction, image)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        image,label = batch
        prediction  = self.forward(image)
        loss = self.loss_fcn(prediction, image)
        self.log("val_loss", loss)
        return loss
   
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=1e-1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        return [optimizer], [scheduler]

if __name__ == "__main__":   

    ## Transforms    
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.RandomResizedCrop(size=(32,32)),
        #transforms.RandomVerticalFlip(p=0.5),
        #transforms.RandomHorizontalFlip(p=0.5),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.ColorJitter(),
        #transforms.RandomRotation(5),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        
    ])

    val_transform   = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]) 
    
    invTrans   = transforms.Compose([
        transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1./0.229, 1./0.224, 1./0.225 ]),
        transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]),        
        torchvision.transforms.ToPILImage()
    ])
    
    ##First create a master loader
    wsi_file, coords_file = LoadFileParameter(sys.argv[1])
    
    coords_file = coords_file[coords_file[ "tumour_label"]==1] ## Only the patches that have tumour in them
    seed_everything(42) 

    callbacks = [
        ModelCheckpoint(
            dirpath='./',
            monitor='val_loss',
            filename="model_AutoEncoder",#.{epoch:02d}-{val_loss:.2f}.h5",
            save_top_k=1,
            mode='min'),
        #EarlyStopping(monitor='val_loss')
    ]

    trainer  = Trainer(gpus=1, max_epochs=10,callbacks=callbacks)
    model    = AutoEncoder()

    data     = DataModule(coords_file, wsi_file, train_transform = train_transform, val_transform = val_transform, batch_size=4)
    trainer.fit(model, data)
    
    ## Testing
    test_dataset       = DataGenerator(coords_file, wsi_file, inference = True,transform=val_transform)
    num_of_predictions = 10
    for n in range(num_of_predictions):
        idx        = np.random.randint(len(coords_file),size=1)[0]

        image     = test_dataset[idx][np.newaxis]
        image_out = model.forward(image)
        image     = invTrans(image.squeeze())
        image_out = invTrans(image_out.squeeze())
        
        plt.subplot(1,2,1)
        plt.imshow(image)
        plt.subplot(1,2,2)
        plt.imshow(image_out)
        plt.show()
