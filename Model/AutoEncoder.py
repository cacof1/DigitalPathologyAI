
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
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
#from piqa import SSIM
from kornia.losses import SSIMLoss
## Module - Dataloaders
from Dataloader.Dataloader import LoadFileParameter, DataModule, DataGenerator, WSIQuery

## Module - Models
from Model.unet import UNet,UNetUpBlock, UNetDownBlock, UNetEncoder
from Model.resnet import ResNet, ResNetResidualBlock, ResNetEncoder, ResNetDecoder

## Main Model
class AutoEncoder(LightningModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config    
        self.encoder = Encoder(backbone = self.config["MODEL"]["Backbone"], config = self.config)
        output_shape = self.encoder(torch.rand((1, self.config["DATA"]["n_channel"], self.config["DATA"]["dim"][0][0], self.config["DATA"]["dim"][0][1] ))).size()
        self.decoder = Decoder(output_shape = output_shape, config = config)
        self.model = nn.Sequential(
            self.encoder,
            self.decoder
        )
        self.loss_fcn = getattr(torch.nn, self.config["MODEL"]["loss_function"])()
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.model(x)
            
    def training_step(self, batch, batch_idx):        
        image,label = batch
        image       = next(iter(image.values())) ## Take the first value in the dictonnary for single zoom
        prediction  = self.forward(image)
        loss        = self.loss_fcn(prediction, image)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        image,label = batch
        image       = next(iter(image.values())) 
        prediction  = self.forward(image)
        loss        = self.loss_fcn(prediction, image)
        self.log("val_loss", loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        image       = batch
        image       = next(iter(image.values()))
        prediction  = self.forward(image)
        return prediction
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=self.config["OPTIMIZER"]["lr"],eps=self.config["OPTIMIZER"]["eps"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config["OPTIMIZER"]["step_size"], gamma=self.config["OPTIMIZER"]["gamma"],verbose=True)
        return [optimizer], [scheduler]
    
class Encoder(LightningModule):
    def __init__(self,backbone=models.densenet121(), config = None) -> None:
        super().__init__()
        if(backbone=="ResNet"):
            self.encoder  = ResNetEncoder(in_channels=config['DATA']['n_channel'], depth=config['MODEL']['depth'], wf =config['MODEL']['wf'])
        elif(backbone=="UNet"):
            self.encoder = UNetEncoder(in_channels=config['DATA']['n_channel'], depth=config['MODEL']['depth'], wf =config['MODEL']['wf'])
        else:
            self.encoder = nn.Sequential(*list(getattr(models, config["MODEL"]["Backbone"])().children())[:-1])

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.encoder(x)

    def predict_step(self, batch, batch_idx):
        image       = batch
        image       = next(iter(image.values()))
        prediction  = self.forward(image)
        return prediction

class Decoder(LightningModule):
    def __init__(self,output_shape=(1,1024,4,4,), config=None) -> None: 
        super().__init__()
        if(config['MODEL']['Backbone']=='ResNet'):
            self.decoder = ResNetDecoder(in_channels = output_shape[1], n_classes =config['DATA']['n_channel'], depth =config['MODEL']['depth'], wf = config['MODEL']['wf'])
        else:
            depth        = int(math.log(128,2) - math.log(output_shape[-1],2))
            in_channels  = output_shape[1]
            out_channels = 0
            self.decoder = nn.ModuleList()
            for i in reversed(range(depth)):
                out_channels = int(2**(math.log(in_channels,2)-1))
                self.decoder.append(UNetUpBlock(in_channels, out_channels))
                in_channels  = out_channels

            self.decoder.append(nn.Conv2d(out_channels, config['DATA']['n_channel'] , kernel_size=1,stride=1))
            self.decoder = nn.Sequential(*self.decoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.decoder(x)    
