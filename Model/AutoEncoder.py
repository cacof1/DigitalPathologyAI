
import matplotlib.pyplot as plt
from pytorch_lightning import LightningDataModule, LightningModule, Trainer,seed_everything
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

## Module - Models
from Model.unet import UNet,UNetUpBlock, UNetDownBlock, UNetEncoder
from Model.resnet import ResNet, ResNetResidualBlock, ResNetEncoder, ResNetDecoder
import monai
## Main Model
class AutoEncoder(LightningModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config    
        self.model = getattr(monai.networks.nets, config['BASEMODEL']['Backbone'])
        self.model = self.model(spatial_dims=2,
                                in_channels=3,
                                out_channels=3,
                                channels=(16, 32, 64),
                                strides=(2, 2, 2))

        print(self.model)
        self.loss_fcn = getattr(torch.nn, self.config["BASEMODEL"]["Loss_Function"])()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.model(x)
            
    def training_step(self, batch, batch_idx):        
        image,label = batch
        prediction  = self.forward(image)
        loss        = self.loss_fcn(prediction, image)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        image,label = batch
        prediction  = self.forward(image)
        loss        = self.loss_fcn(prediction, image)

        if(batch_idx<3):
            im          = np.swapaxes(prediction.cpu()[0],0,-1)
            print(im)
            print(im.dtype)
            plt.imshow(im)
            plt.show()
        self.log("val_loss", loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        image       = batch
        prediction  = self.forward(image)
        im          = np.swapaxes(prediction.cpu().numpy()[0],0,-1)
        print(im)
        print(im.dtype)
        
        plt.imshow(im)
        plt.show()
        return prediction
    
    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.config['OPTIMIZER']['Algorithm'])
        optimizer = optimizer(self.parameters(),
                              lr=self.config["OPTIMIZER"]["lr"],
                              eps=self.config["OPTIMIZER"]["eps"],
                              betas=(0.9, 0.999),
                              weight_decay=self.config['REGULARIZATION']['Weight_Decay'])

        if self.config['SCHEDULER']['Type'] == 'cosine_warmup':
            n_steps_per_epoch = self.config['DATA']['N_Training_Examples'] // self.config['BASEMODEL']['Batch_Size']
            total_steps = n_steps_per_epoch * self.config['ADVANCEDMODEL']['Max_Epochs']
            warmup_steps = self.config['SCHEDULER']['Cos_Warmup_Epochs'] * n_steps_per_epoch

            sched = transformers.optimization.get_cosine_schedule_with_warmup(optimizer,
                                                                              num_warmup_steps=warmup_steps,
                                                                              num_training_steps=total_steps,
                                                                              num_cycles=0.5)  # default lr->0.                                                                                                    

            scheduler = {'scheduler': sched,
                         'interval': 'step',
                         'frequency': 1}

        elif self.config['SCHEDULER']['Type'] == 'stepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                        step_size=self.config["SCHEDULER"]["Lin_Step_Size"],
                                                        gamma=self.config["SCHEDULER"][
                                                            "Lin_Gamma"])  # step size 5, gamma =0.5                                                                                                               
        return ([optimizer], [scheduler])
