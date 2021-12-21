import matplotlib.pyplot as plt
from pytorch_lightning import LightningDataModule, LightningModule, Trainer,seed_everything
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import pandas as pd
import cv2,math

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
#from piqa import SSIM
from kornia.losses import SSIMLoss
## Module - Dataloaders
from Dataloader.Dataloader import LoadFileParameter, DataModule, DataGenerator, WSIQuery

## Module - Models
from Model.unet import UNet,UNetUpBlock
from Model.resnet import ResNet, ResNetResidualBlock, ResNetEncoder

## Main Model
class AutoEncoder(LightningModule):
    def __init__(self,backbone=models.densenet169(),n_classes =3,dim = (128,128)) -> None:
        super().__init__()
        backbone     = nn.Sequential(*list(backbone.children())[:-1]) ## remove classifier
        output_shape = backbone(torch.rand((1, n_classes, dim[0], dim[1]))).size()
        depth        = int(math.log(128,2) - math.log(output_shape[-1],2))
        in_channels  = output_shape[1]
        self.encoder = backbone#UNet(in_channels=3, n_classes =3, depth= depth, wf =wf).encoder
        self.decoder = nn.ModuleList()
        for i in reversed(range(depth)):
            out_channels = int(2**(math.log(in_channels,2)-1))
            self.decoder.append(UNetUpBlock(in_channels, out_channels))
            in_channels  = out_channels

        self.decoder.append(nn.Conv2d(out_channels, n_classes, kernel_size=1,stride=1))
        self.decoder = nn.Sequential(*self.decoder)
        self.model = nn.Sequential(
            self.encoder,
            self.decoder
        )
        self.loss_fcn = torch.nn.MSELoss()
        #self.loss_fcn_2 = torch.nn.L1Loss()        
        #self.loss_fcn  = SSIMLoss(5)
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
        loss = self.loss_fcn(prediction, image)
        self.log("val_loss", loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        image = batch
        prediction  = self.forward(image)
        return prediction
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=1e-3,eps=1e-7)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1,verbose=True)
        return [optimizer], [scheduler]
    
