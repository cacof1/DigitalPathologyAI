import matplotlib.pyplot as plt
from pytorch_lightning import LightningDataModule, LightningModule
import numpy as np
import torch
from torch import nn
from collections import Counter
import torchvision
from torchvision import datasets, models, transforms
from torchvision import transforms
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import LightningDataModule, LightningModule, Trainer,seed_everything
from torchsummary import summary
import sys
import torchio as tio
import sklearn
from pytorch_lightning import loggers as pl_loggers
import torchmetrics

class MixModel(LightningModule):
    def __init__(self, model_dict, loss_fcn = torch.nn.BCEWithLogitsLoss() ):
        super().__init__()
        self.model_dict = model_dict
        self.classifier = nn.Sequential(
            nn.LazyLinear(128),
            nn.LazyLinear(1)
        )
        self.accuracy = torchmetrics.AUC(reorder=True)
        self.loss_fcn = loss_fcn

    def forward(self, data_dict):
        features = torch.cat([model_dict[key](data_dict[key]) for key in self.model_dict.keys()], dim=1) 
        return self.classifier(features)

    def training_step(self, batch,batch_idx):
        data_dict, label = batch
        prediction  = self.forward(data_dict)
        loss = self.loss_fcn(prediction.squeeze(dim=1),label)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch,batch_idx):
        data_dict, label = batch
        prediction  = self.forward(data_dict)
        loss = self.loss_fcn(prediction.squeeze(dim=1),label)
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return [optimizer], [scheduler]

