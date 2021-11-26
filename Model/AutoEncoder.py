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
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

## Module - Dataloaders
from Dataloader.Dataloader import LoadFileParameter, DataModule, DataGenerator, WSIQuery

## Module - Models
from Model.unet import UNet, Decoder
from Model.simplemodel import SimpleNet
from Model.resnet import ResNet, ResNetResidualBlock, ResNetEncoder

## Main Model
class AutoEncoder(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        wf = 5
        depth = 5 
        #self.model   = SimpleNet(in_channels=3, n_classes = 3, depth=depth,wf=wf)
        self.model = UNet(in_channels=3, n_classes =3, depth= depth, wf =wf) 
        summary(self.model.to('cuda'), (3,96,96))
        
        self.loss_fcn = torch.nn.MSELoss()
        #self.loss_fcn = torch.nn.L1Loss(reduction="sum")        

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
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

        """
        if(batch_idx==0):
            n = 10
            plt.figure(figsize=(20, 4))
            for i in range(n):
                test = image.cpu().numpy()[i]
                test = test.transpose((1,2,0))
                
                ax = plt.subplot(2, n, i + 1)
                plt.imshow(test)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                test2 = prediction.cpu().numpy()[i]
                test2 = test2.transpose((1,2,0))

                ax = plt.subplot(2, n, i + 1 + n)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)                
                plt.imshow(test2)
            plt.show()
        """
        loss = self.loss_fcn(prediction, image)
        self.log("val_loss", loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        image = batch
        prediction  = self.forward(image)
        """
        if(batch_idx==0):
            n = 10
            plt.figure(figsize=(20, 4))
            for i in range(n):
                test = image.cpu().numpy()[i]
                test = test.transpose((1,2,0))
                
                ax = plt.subplot(2, n, i + 1)
                plt.imshow(test)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                test2 = prediction.cpu().numpy()[i]
                test2 = test2.transpose((1,2,0))

                ax = plt.subplot(2, n, i + 1 + n)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)                
                plt.imshow(test2)
            plt.show()        
        """
        return prediction
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=1e-3,eps=1e-7)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1,verbose=True)
        return [optimizer], [scheduler]
    
